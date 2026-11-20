#!/usr/bin/env python3
"""Measure convergence of signed native and model derivatives at three surface speeds."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin
import torch.nn.functional as F

from audit_supermimic_paired_racket_fixture import signatures, verify_initial_intervention, common_native_twist, verify_common_twist, verify_direction
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick


def derivative(plus, minus, epsilon):
    if epsilon <= 0:
        raise ValueError('Positive intervention amplitude required')
    return (plus - minus) / (2 * epsilon)


def vector_summary(reference, estimate):
    difference = estimate - reference
    valid = reference.norm(dim=-1) > 1.e-7
    result = dict(rows=len(reference), supported=int(valid.sum()),
                  component_rmse=float(difference.square().mean().sqrt()))
    if valid.any():
        ref, pred = reference[valid], estimate[valid]
        cosine = F.cosine_similarity(ref, pred)
        ratio = pred.norm(dim=-1) / ref.norm(dim=-1)
        result.update(cosine_median=float(cosine.median()), cosine_p10=float(cosine.quantile(.1)),
                      gain_median=float(ratio.median()), gain_p10=float(ratio.quantile(.1)))
    return result


def read_trial(directory, seed, branch, expected, settings, prior, design=None, *, material=None):
    path = directory / branch / f'spec_{seed}' / 'trace.pt'
    trial = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
    verify_direction(trial,design)
    tr = trial['raw']['substep_trace']
    t = tr['tensors']
    v = trial['velocity_intervention']
    if (v['branch'] != branch or trial['seed'] != seed or v['epsilon_mps'] != expected
            or tr['count'] != prior.steps + 1 or tr['physics_dt_s'] != prior.sub_dt):
        raise ValueError(f'Unexpected physical control interval in {path}')
    if 'sdk_cached_initial_twist' in v:
        verify_initial_intervention(tr,v)
    sign = {'minus': -1, 'plus': 1, 'nominal_a': 0, 'nominal_b': 0}[branch]
    torch.testing.assert_close(v['delta_world'], sign*expected*v['direction_world'], rtol=0, atol=1.e-7)
    names = ('racket_position_w', 'racket_quaternion_w', 'racket_linear_velocity_w', 'racket_angular_velocity_w')
    ball = t['shuttle_state_w'][0]
    state = torch.cat((ball[:, :3], ball[:, 7:10]), dim=-1)
    origin = fixture_court_origin(trial,state[:,:3],design)
    state[:,:3] -= origin
    stream = [t[k] for k in names]
    stream[0] = stream[0] - origin
    out, _ = tick(prior, state, [x[0] for x in stream], [x[-1] for x in stream],
                  settings['fitted'], settings['solver']['pair_contact_offset_m'],
                  None if material is not None else torch.zeros(len(ball), dtype=torch.bool),
                  measured_path=[x[:-1] for x in stream],
                  racket_material=material, compound_racket=material is not None,
                  initial_spin=ball[:, 10:13] if material is not None else None)
    common = {k: v[k] for k in ('nominal_shuttle', 'joint_targets', 'direction_world', 'nominal_root_com_velocity')}
    common.update(court_origin=origin,ball_state=ball, racket_position=t[names[0]][0], racket_rotation=t[names[1]][0],
                  parent_lane=trial['parent_lane'])
    if 'sdk_cached_initial_twist' in v:
        common['native_write'] = common_native_twist(v)
    if trial['parent_family'] != seed % 16:
        raise ValueError('Launch family mismatch')
    return common, signatures(tr), t['shuttle_state_w'][-1, :, 7:10], out[:, 3:], sha(path)


def read_material(wide, prov, material_path, readout_path, calibration_path):
    material = None
    if material_path:
        from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract
        from fit_supermimic_racket_material_response import material_table
        teacher_path = Path(prov['args'][prov['args'].index('--teacher')+1])
        if sha(teacher_path)!=prov['teacher_sha256']:
            raise ValueError('Changed fixture teacher asset')
        asset=torch.load(teacher_path,mmap=True,weights_only=True)['task_contract']['asset']['sha256']
        contract=validate_contract(json.loads(material_path.read_text()),asset)
        material=torch.tensor(contract['effective'])
        if readout_path:
            fit=json.loads(readout_path.read_text())
            if (fit['calibration_sha256']!=sha(calibration_path) or fit['material_sha256']!=sha(material_path)
                    or fit['split_hashes']['train']!=sha(wide/'train.json') or fit['final_holdout_opened']):
                raise ValueError('Incompatible predictive coefficient calibration')
            strings=torch.tensor([n.startswith('Collider/strip_') for n in contract['component_names']])
            material=material_table(material,strings,torch.tensor(fit['selected']))
    return material


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wide', 'scales', 'wide-audit', 'calibration', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--compound-material',type=Path,help='Use the multi-shape solve and propagate incoming native spin')
    parser.add_argument('--material-readout',type=Path,help='Optional two-material predictive coefficients fitted on calibration families')
    args = parser.parse_args()
    if args.material_readout and not args.compound_material:
        parser.error('A material readout requires the native source material contract')
    torch.set_num_threads(3)
    plan = json.loads((args.scales/'design.json').read_text())
    design = json.loads((args.wide/'design.json').read_text())
    audit = json.loads(args.wide_audit.read_text())
    if (plan['parent_design_sha256'] != sha(args.wide/'design.json')
            or audit['design_sha256'] != plan['parent_design_sha256'] or audit['split'] not in ('train', 'development')
            or plan['epsilon_mps'] != [.025, .0125] or plan['wide_epsilon_mps'] != .05):
        raise ValueError('Expected the three registered amplitudes and the calibration families')
    cfg = json.loads(args.calibration.read_text())
    prior = BallImpulsePriorV1().eval()
    directories = [args.wide, args.scales/'half', args.scales/'quarter']
    epsilons = [plan['wide_epsilon_mps'], *plan['epsilon_mps']]
    prov = json.loads((args.wide/'nominal_a'/'provenance.json').read_text())
    material = read_material(args.wide,prov,args.compound_material,args.material_readout,args.calibration)
    for directory in directories:
        for branch in ('plus','minus'):
            check = json.loads((directory/branch/'provenance.json').read_text())
            for k in ('teacher_sha256','collector_sha256','trace_recorder_sha256','seed','batches','lanes'):
                if check[k] != prov[k]:
                    raise ValueError(f'Different {k} in {directory}/{branch}')
    by_case = {(r['seed'],r['lane']): r for r in audit['samples']}
    keys = ('native_wide', 'native_half', 'native_quarter', 'model_wide', 'model_half', 'model_quarter',
            'native_left', 'native_right')
    output = {k: [] for k in keys}
    groups = {k: [] for k in ('all','wide_stable','all_scales_stable','wide_weak_and_all_scales_stable',
                                'native_convergent_stable','stable_without_native_convergence')}
    hashes = {}
    for sk, part in design['assignments'].items():
        if part != audit['split']:
            continue
        seed = int(sk)
        common, nominal, base, _, digest = read_trial(args.wide,seed,'nominal_a',epsilons[0],cfg,prior,design,material=material)
        if digest != audit['archives'][str(args.wide/'nominal_a'/f'spec_{seed}'/'trace.pt')]:
            raise ValueError('Nominal panel changed')
        stable = torch.ones(len(base),dtype=torch.bool)
        for name,eps,dir_ in zip(('wide','half','quarter'),epsilons,directories):
            signed = {}
            for branch in ('minus','plus'):
                ref, events, native, model, digest = read_trial(dir_,seed,branch,eps,cfg,prior,design,material=material)
                if design.get('initial_racket_twist'):
                    verify_common_twist(common['native_write'],ref['native_write'])
                if any(not torch.equal(v,ref[k]) for k,v in common.items() if k!='native_write'):
                    raise ValueError(f'Initial state or command mismatch in {dir_}/{branch}/{seed}')
                signature = torch.tensor([a == b for a,b in zip(events,nominal)])
                stable &= signature
                path = str(dir_/branch/f'spec_{seed}'/'trace.pt')
                if name=='wide' and digest!=audit['archives'][path]:
                    raise ValueError('Wide panel changed')
                hashes[path]=digest
                signed[branch]=(native,model)
            for i,label in enumerate(('native','model')):
                output[label+'_'+name].append(derivative(signed['plus'][i],signed['minus'][i],eps))
            if name=='quarter':
                output['native_left'].append((base-signed['minus'][0])/eps)
                output['native_right'].append((signed['plus'][0]-base)/eps)
        wide_stable = torch.tensor([by_case[seed,l]['kind']=='racket_stable' for l in range(len(base))])
        weak = F.cosine_similarity(output['native_wide'][-1],output['model_wide'][-1])<.5
        # A Newton derivative has to describe both sides of the nominal response.
        # Define this diagnostic with native signals only and retain the original panel.
        reproducible = stable & wide_stable
        for x, y in (('native_wide', 'native_half'), ('native_half', 'native_quarter'),
                     ('native_left','native_right')):
            left, right = output[x][-1], output[y][-1]
            gain = right.norm(dim=-1)/left.norm(dim=-1).clamp_min(1.e-12)
            reproducible &= (F.cosine_similarity(left,right)>=.8) & (gain>=.5) & (gain<=2.)
        reproducible &= output['native_wide'][-1].norm(dim=-1)*2*epsilons[0]>.01
        groups['native_convergent_stable'].append(reproducible)
        groups['stable_without_native_convergence'].append(stable & wide_stable & ~reproducible)
        groups['all'].append(torch.ones_like(stable))
        groups['wide_stable'].append(wide_stable)
        groups['all_scales_stable'].append(wide_stable & stable)
        groups['wide_weak_and_all_scales_stable'].append(wide_stable & stable & weak)
    data={k:torch.cat(v) for k,v in output.items()}
    if len(data['native_wide'])!=len(by_case):
        raise ValueError('Incomplete training split')
    results={}
    for group, masks in groups.items():
        selected=torch.cat(masks)
        if not selected.any():
            results[group]=dict(rows=0)
            continue
        d={k:v[selected] for k,v in data.items()}
        comparisons = {'native_wide_to_half':('native_wide','native_half'),
                       'native_half_to_quarter':('native_half','native_quarter'),
                       'quarter_native_left_to_right':('native_left','native_right')}
        for tag in ('wide','half','quarter'):
            comparisons['model_vs_native_'+tag]=('native_'+tag,'model_'+tag)
        results[group]={k:vector_summary(d[ref],d[pred]) for k,(ref,pred) in comparisons.items()}
    with args.output.open('x') as f:
        json.dump(dict(material_sha256=sha(args.compound_material) if args.compound_material else None,
                       coefficient_fit_sha256=sha(args.material_readout) if args.material_readout else None,
                       calibration_sha256=sha(args.calibration),wide_audit_sha256=sha(args.wide_audit),
                       scale_plan_sha256=sha(args.scales/'design.json'),physical_archives=hashes,
                       units='change in ball m/s per 1 m/s requested native surface perturbation',
                       groups=results,case_ids=[[seed,l] for seed in sorted({key[0] for key in by_case})
                                             for l in range(design['lanes'])],
                       incident_labels={k:torch.cat(v).tolist() for k,v in groups.items()
                                        if k in ('all_scales_stable','native_convergent_stable')},
                       split=audit['split'],policy_updates=0,contact_bptt_admitted=False),f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
