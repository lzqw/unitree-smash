#!/usr/bin/env python3
"""Score signed native moving-racket trials separately from end-to-end policy perturbations."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin

from audit_supermimic_compound_impact_teacher import predict
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def signatures(trace):
    t=trace['tensors']
    nr=t['racket_force_w'][1:].norm(dim=-1)>.001
    fr=t['racket_friction_force_w'][1:].norm(dim=-1)>.001
    grd=t['ground_force_w'][1:].norm(dim=-1)>.001
    net=t['net_force_matrix_w'][1:].reshape(*nr.shape,-1,3).norm(dim=-1).amax(-1)>.001
    result=[]
    for lane in range(nr.shape[1]):
        frames=(nr[:,lane]|fr[:,lane]).nonzero().flatten().tolist()
        points=[]
        for i in frames:
            m=trace['manifold_frames'][i+1];f=trace['racket_friction_frames'][i+1]
            points.append((int(((m['lane']==lane)&(m['filter_index']==0)&(m['normal_force_n'].abs()>.001)).sum()),
                           int(((f['lane']==lane)&(f['force_w_n'].norm(dim=-1)>.001)).sum())))
        result.append((bool(grd[:,lane].any()),bool(net[:,lane].any()),bool(frames),tuple(frames),tuple(points)))
    return result


def inferred_support(trace, geometry):
    """Closest analytic shape to each native loaded point, with explicit ambiguity.

    PhysX supplies a body/filter ID and the world contact point here. The shape
    correspondence is inferred from geometry and is never a native shape ID.
    """
    from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
    from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse
    t=trace['tensors']; by_lane=[[] for _ in range(t['racket_force_w'].shape[1])]
    g=geometry
    for frame,m in enumerate(trace['manifold_frames'][1:],1):
        keep=(m['filter_index']==0)&(m['normal_force_n'].abs()>.001)
        if not keep.any():continue
        lane=m['lane'][keep]
        point=m['point_w'][keep]
        local=_quat_rotate_inverse(t['racket_quaternion_w'][frame-1,lane],point-t['racket_position_w'][frame-1,lane])
        gap=torch.cat([box_gap(local,g.racket_box_centers,g.racket_box_halves),
                       round_gap(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,
                                 g.racket_round_radii,g.racket_round_capsule)],-1).abs()
        distance, shape=gap.topk(2,largest=False)
        for l in lane.unique().tolist():
            selected=lane==l
            # Marks a correspondence unresolved when the nearest surfaces are
            # closer together than the measured point's own surface discrepancy.
            confident=(distance[selected,1]-distance[selected,0])>distance[selected,0].clamp_min(1e-5)
            ids=shape[selected,0].tolist()
            by_lane[l].append((frame,tuple(sorted(ids)),bool(confident.all())))
    return by_lane


def classify(rows):
    if rows[0]!=rows[1]:return 'nominal_mismatch'
    if any(r[:3]!=rows[0][:3] for r in rows[2:]):return 'object_switch'
    if any(rows[0][:2]):return 'nonracket_contact'
    if not rows[0][2]:return 'no_contact'
    if any(r[3]!=rows[0][3] for r in rows[2:]):return 'toi_switch'
    if any(r[4]!=rows[0][4] for r in rows[2:]):return 'manifold_switch'
    return 'racket_stable'


def verify_initial_intervention(trace, intervention):
    """Verify the native velocity write and the teacher tensor used at the first substep."""
    kind = trace['contact_metadata'].get('initial_racket_twist_source')
    if kind != 'direct_native_link_velocity_after_intervention':
        raise ValueError('Missing direct native racket velocity readback')
    raw = intervention['native_velocity_readback']
    before, after = raw['before'], raw['after']
    change = intervention['delta_world']
    torch.testing.assert_close(after['root_com'][:, :3]-before['root_com'][:, :3], change, atol=2.e-5, rtol=0)
    torch.testing.assert_close(after['root_com'][:, 3:], before['root_com'][:, 3:], atol=2.e-5, rtol=0)
    for key, field in [('racket_linear_velocity_w','racket_link'),('racket_angular_velocity_w','racket_angular')]:
        torch.testing.assert_close(trace['tensors'][key][0], after[field], atol=0, rtol=0)


def common_native_twist(intervention):
    # Compare branches at matching native forward boundaries. Even a zero
    # root write can refresh the derived link velocity at fixed pose.
    raw = intervention['native_velocity_readback']
    result = {'before_'+k: v for k,v in raw['before'].items()}
    result.update({'after_'+k: raw['after'][k] for k in ('racket_angular','dof')})
    result['after_racket_reference'] = raw['after']['racket_link']-intervention['delta_world']
    return result


def verify_common_twist(reference, other):
    for key, value in reference.items():
        tolerance = 2.e-5 if key == 'after_racket_reference' else 0.
        torch.testing.assert_close(value, other[key], atol=tolerance, rtol=0)


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--split',choices=('train','development','final','previously_explored'))
    for name in ('collection','material','calibration','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    a=cli.parse_args();torch.set_num_threads(3)
    prov={b:json.loads((a.collection/b/'provenance.json').read_text()) for b in BRANCHES}
    source=prov['nominal_a']
    for key in ('teacher_sha256','collector_sha256','trace_recorder_sha256','seed','batches','lanes'):
        if not all(v[key]==source[key] for v in prov.values()):raise ValueError('Fixture sources differ: '+key)
    teacher_path=Path(source['args'][source['args'].index('--teacher')+1])
    if sha(teacher_path)!=source['teacher_sha256']:raise ValueError('Teacher binding differs')
    teacher=torch.load(teacher_path,map_location='cpu',weights_only=True)
    material=validate_contract(json.loads(a.material.read_text()),teacher['task_contract']['asset']['sha256'])
    material=torch.tensor(material['effective'])
    cfg=json.loads(a.calibration.read_text())
    prior=BallImpulsePriorV1().eval()
    physics=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    design=None
    if (a.collection/'design.json').is_file():
        design=json.loads((a.collection/'design.json').read_text())
        if not a.split:cli.error('Select one preregistered fixture family split')
        if any(design[k]!=source[k] for k in ('seed','lanes','batches')) or design['source_teacher_sha256']!=source['teacher_sha256']:
            raise ValueError('Design and collection differ')
    elif a.split:cli.error('Cannot score a split without the collection design')
    hashes={};results=[]
    for seed in range(source['seed'],source['seed']+source['batches']):
        if design and design['assignments'][str(seed)]!=a.split:continue
        case={};modes={};inputs={};parts={}
        for b in BRANCHES:
            path=a.collection/b/f'spec_{seed}'/'trace.pt'
            hashes[str(path)]=sha(path)
            trial=torch.load(path,mmap=True,map_location='cpu',weights_only=True)
            if trial.get('schema')!='native-controlled-contact-trial-v1':
                raise ValueError('A loaded native fixture is required for impact labels')
            tr=trial['raw']['substep_trace'];t=tr['tensors'];v=trial['velocity_intervention']
            if v['branch']!=b or trial['seed']!=seed or tr['count']!=prior.steps+1 or tr['physics_dt_s']!=prior.sub_dt:
                raise ValueError('Invalid physical branch or clock')
            if design and design.get('initial_racket_twist'):
                verify_initial_intervention(tr, v)
            sign={'nominal_a':0,'nominal_b':0,'minus':-1,'plus':1}[b]
            torch.testing.assert_close(v['delta_world'],sign*v['epsilon_mps']*v['direction_world'],atol=1e-7,rtol=0)
            inputs[b]=dict(nominal_shuttle=v['nominal_shuttle'],joint_targets=v['joint_targets'],
                           direction=v['direction_world'],nominal_root_velocity=v['nominal_root_com_velocity'],
                           initial_shuttle=t['shuttle_state_w'][0],initial_racket_position=t['racket_position_w'][0],
                           initial_racket_rotation=t['racket_quaternion_w'][0],parent=trial['parent_lane'],
                           epsilon=torch.tensor(v['epsilon_mps']))
            if design and design.get('initial_racket_twist'):
                inputs[b]['native_write'] = common_native_twist(v)
            p=t['shuttle_state_w'][0];ball=torch.cat((p[:,:3],p[:,7:10]),-1)
            names=('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
            origin=fixture_court_origin(trial,ball[:,:3],design)
            inputs[b]['court_origin']=origin
            ball[:,:3]-=origin
            stream=[t[k] for k in names]
            stream[0]=stream[0]-origin
            pred,_=tick(prior,ball,[x[0] for x in stream],[x[-1] for x in stream],
                cfg['fitted'],cfg['solver']['pair_contact_offset_m'],torch.zeros(len(p),dtype=torch.bool),measured_path=[x[:-1] for x in stream])
            if design and (trial['parent_family']!=seed%16 or
                    trial['parent_family'] not in design[{'train':'calibration_families','development':'development_families',
                    'final':'reserved_final_families','previously_explored':'previously_explored_families'}[a.split]]):
                raise ValueError('Native parent family does not match the predeclared split')
            modes[b]=signatures(tr)
            parts[b]=inferred_support(tr,prior.preview)
            case[b]=dict(native=t['shuttle_state_w'][-1,:,7:10],native_racket_analytic_rollout=pred[:,3:],
                         racket_velocity_next=t['racket_linear_velocity_w'][1])
            for name,use in [('deepest',False),('compound',True)]:
                terms=predict(tr,list(range(len(p))),prior,material,cfg,compound=use)
                case[b][name+'_racket_impulse_teacher']=(terms['normal']+terms['tangent']).sum(0)
            case[b]['native_racket_impulse']=(t['racket_force_w'][1:]+t['racket_friction_force_w'][1:]).sum(0)*(prior.sub_dt/physics['mass_kg'])
        base=inputs['nominal_a']
        for b in BRANCHES[1:]:
            for k,v in base.items():
                if k=='native_write':
                    verify_common_twist(v, inputs[b][k])
                    continue
                if not torch.equal(v,inputs[b][k]):
                    raise ValueError(f'Fixture {seed}, {b}: common physical input differs: {k}')
        for lane in range(source['lanes']):
            delta={key:(case['plus'][key]-case['minus'][key])[lane].tolist() for key in case['plus']}
            repeat={key:(case['nominal_a'][key]-case['nominal_b'][key])[lane].tolist() for key in case['plus']}
            rows=[modes[b][lane] for b in BRANCHES]
            support=[parts[b][lane] for b in BRANCHES]
            confident=all(branch and all(entry[2] for entry in branch) for branch in support)
            match=confident and all(branch==support[0] for branch in support[1:])
            results.append(dict(seed=seed,lane=lane,kind=classify(rows),inferred_support_match=match,
                                inferred_support_confident=confident,delta=delta,repeat=repeat,
                                requested_surface_delta_mps=(2*base['epsilon']*base['direction'][lane]).tolist()))
    groups={}
    for label in ['all','racket_stable_same_inferred_shape']+sorted(set(r['kind'] for r in results)):
        group=[r for r in results if label=='all' or r['kind']==label]
        if label=='racket_stable_same_inferred_shape':
            group=[r for r in results if r['kind']=='racket_stable' and r['inferred_support_match']]
        if not group:continue
        arr=lambda key,src:torch.tensor([r[src][key] for r in group])
        g={}
        for model,ref in [('native_racket_analytic_rollout','native'),('deepest_racket_impulse_teacher','native_racket_impulse'),('compound_racket_impulse_teacher','native_racket_impulse')]:
            g[model]=response_metrics(arr(ref,'delta'),arr(model,'delta'),arr(ref,'repeat'))
        g['native_response_over_0p01_mps']=int((arr('native','delta').norm(dim=-1)>.01).sum())
        g['native_repeat_max_mps']=float(arr('native','repeat').norm(dim=-1).max())
        g['delivered_surface_gain_median']=float((arr('racket_velocity_next','delta').norm(dim=-1)/
            torch.tensor([r['requested_surface_delta_mps'] for r in group]).norm(dim=-1)).median())
        groups[label]=g
    out=dict(scope='Native paired root velocity fixture; native racket teacher only; recorded ball states used only by impulse teachers',
             split=a.split,design_sha256=sha(a.collection/'design.json') if design else None,
             trials=len(results),native_branches=len(results)*4,archives=hashes,groups=groups,samples=results,
             policy_training_admitted=False,actor_updates=0,critic_updates=0)
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
