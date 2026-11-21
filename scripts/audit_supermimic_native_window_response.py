#!/usr/bin/env python3
"""Audit two ball rollouts over a continuous window of recorded G07 commands and racket motion."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, PhysicalWorldV2, predicted_racket_path, racket, sha, tick
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_native_manifold_groups import classify, signatures
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES, perturb_request, select_substep_lanes
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def complete_window(raw, spec, steps):
    lanes = spec['lanes']
    indices = list(range(spec['action_tick'], spec['tick'] + 1))
    if spec['capture_substep_ticks'] != indices or len(raw['step_sequence']) != len(indices):
        raise ValueError('Missing control interval in executable probe')
    if set(raw['earlier_substep_traces']) != set(indices[:-1]):
        raise ValueError('Missing pre-terminal substep trace')
    traces = [raw['earlier_substep_traces'][i] for i in indices[:-1]]
    traces.append(select_substep_lanes(raw['substep_trace'], lanes))
    previous = None
    for trace, control in zip(traces, raw['step_sequence']):
        frames = trace['tensors']['physics_frame']
        if (trace['original_lanes'] != lanes or len(frames) != steps+1 or
                not (frames.diff() == 1).all() or len(trace['manifold_frames']) != len(frames) or
                len(trace['racket_friction_frames']) != len(frames)):
            raise ValueError('Native sample clock, case identities or contact records differ')
        shuttle = trace['tensors']['shuttle_state_w']
        if previous is not None:
            if previous[0] != int(frames[0]):
                raise ValueError('Discontinuous native physics clock')
            torch.testing.assert_close(shuttle[0], previous[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(control['next_state'][lanes,49:],
                                   torch.cat((shuttle[-1,:,:3],shuttle[-1,:,7:10]), -1))
        if control['done'][lanes].any() or not control['ball_present'][lanes].all():
            raise ValueError('Reset or absent shuttle in recorded flight window')
        previous = (int(frames[-1]),shuttle[-1])
    return traces


def window_signature(sequence, steps):
    """Keep the physics-frame index and point counts across all control intervals."""
    loaded = [(i*steps + t, c) for i, row in enumerate(sequence) for t,c in zip(row[3],row[4])]
    return (any(row[0] for row in sequence), any(row[1] for row in sequence), bool(loaded),
            tuple(t for t,c in loaded), tuple(c for t,c in loaded))


@torch.no_grad()
def measure(run, body, prior, material, cfg, launch, *, split_racket_path=False, paths=None):
    spec,raw = run['spec'],run['raw']
    lanes = spec['lanes']
    ts = complete_window(raw,spec,prior.steps)
    initial = raw['start']['before']['physical']
    state = encode_physical(initial)[lanes]
    previous, feet, elapsed, present = [initial[k][lanes] for k in
                                        ('final_action','foot_contact','elapsed_s','ball_present')]
    if not present.all():
        raise ValueError('Initial shuttle not yet launched')
    first = ts[0]['tensors']['shuttle_state_w'][0]
    torch.testing.assert_close(state[:,49:],torch.cat((first[:,:3],first[:,7:10]),-1))
    teacher_ball = state[:,49:]
    teacher_spin = body_spin = first[:,10:13]
    hybrids = ({mode:(teacher_ball,teacher_spin) for mode in ('native_pose','native_twist','native_endpoints')}
               if split_racket_path else {})
    timeline = []
    for trace,control in zip(ts,raw['step_sequence']):
        t = trace['tensors']
        if abs(trace['physics_dt_s']-prior.sub_dt) > 1e-12:
            raise ValueError('Different physics step')
        targets = control['final_joint_targets'][lanes]
        robot,contact,_ = body(state,targets,previous,feet.float(),present,elapsed,launch)
        start,end = racket(state,body.geometry),racket(robot,body.geometry)
        settings = dict(racket_material=material,compound_racket=True,return_spin=True)
        args = (start,end,cfg['fitted'],cfg['solver']['pair_contact_offset_m'])
        native_path = [t[k][:-1] for k in ('racket_position_w','racket_quaternion_w',
                                            'racket_linear_velocity_w','racket_angular_velocity_w')]
        teacher_ball,teacher_hit,teacher_spin = tick(prior,teacher_ball,*args,initial_spin=teacher_spin,
                                                    measured_path=native_path,**settings)
        body_ball,body_hit,body_spin = tick(prior,state[:,49:],*args,initial_spin=body_spin,**settings)
        if hybrids or paths is not None:
            interpolation=dict(rotation_hermite=cfg['fitted'].get('racket_rotation_hermite',False))
            estimate = predicted_racket_path(prior,start,end,**interpolation)
        if paths is not None:
            paths.append(dict(native=native_path, body=estimate))
        if hybrids:
            keys=('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
            native_endpoints=predicted_racket_path(prior,[t[k][0] for k in keys],[t[k][-1] for k in keys],**interpolation)
            for mode,path in [('native_pose',(*native_path[:2],*estimate[2:])),
                              ('native_twist',(*estimate[:2],*native_path[2:])),
                              ('native_endpoints',native_endpoints)]:
                incoming,spin=hybrids[mode]
                new_ball,_,new_spin=tick(prior,incoming,*args,initial_spin=spin,measured_path=path,**settings)
                hybrids[mode]=new_ball,new_spin
        state = torch.cat((robot[:,:49],body_ball),-1)
        previous = control['final_action'][lanes]
        feet = contact.sigmoid()
        elapsed = elapsed + prior.dt
        timeline.append(dict(native=t['shuttle_state_w'][-1,:,7:10],measured_racket=teacher_ball[:,3:],
                             body_racket=body_ball[:,3:], body_racket_linear=end[2], body_racket_angular=end[3],
                             native_racket_linear=t['racket_linear_velocity_w'][-1],
                             native_racket_angular=t['racket_angular_velocity_w'][-1], target=targets,
                             final_action=previous, measured_racket_hit=teacher_hit,body_racket_hit=body_hit,
                             topology=signatures(trace)))
        timeline[-1].update({name:b[:,3:] for name,(b,_) in hybrids.items()})
    return timeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('collection','body','calibration','material','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--exchange-reference',type=Path)
    p.add_argument('--exchange-convergence',type=Path)
    a = p.parse_args()
    if bool(a.exchange_reference) != bool(a.exchange_convergence):
        p.error('Trajectory exchange requires reference and convergence reports')
    torch.set_num_threads(2)
    status_path = a.collection/'status.json'
    status = json.loads(status_path.read_text())
    if status['status'] != 'COLLECTION_FINISHED' or len(status['completed']) != len(BRANCHES):
        raise ValueError('A full four-branch collection is required')
    names = {row['spec'] for row in status['completed']}
    if len(names) != 1 or {row['branch'] for row in status['completed']} != set(BRANCHES):
        raise ValueError('Mismatched probe family')
    folder = a.collection/names.pop()
    runs = {b:_load_branch(folder,b) for b in BRANCHES}
    _same_contract(runs)
    for row in status['completed']:
        if sha(folder/row['branch']/'native_probe.pt') != row['probe_sha256']:
            raise ValueError('Native branch hash changed')
    for branch, execution in runs.items():
        raw = execution['raw']
        torch.testing.assert_close(raw['request_after'], perturb_request(raw['request_before'], execution['spec'], branch),
                                   rtol=0, atol=0)
    run = runs['nominal_a']
    task,spec = run['task_contract'],run['spec']
    ck = torch.load(a.body,map_location='cpu',weights_only=True)
    if ck['source_manifest']['task_contract'] != task or set(spec['lanes']) & set(ck['calibration']['train_lanes']):
        raise ValueError('Body contract or held-out lanes differ')
    suite = Path(task['suite']['path'])
    if sha(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite changed')
    launches = torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps']
                             for x in json.loads(suite.read_text())['cases']])[spec['lanes']]
    material = torch.tensor(validate_contract(json.loads(a.material.read_text()),task['asset']['sha256'])['effective'])
    cfg = json.loads(a.calibration.read_text())
    prior = BallImpulsePriorV1().eval()
    body = PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict(ck['model'])
    if a.exchange_reference:
        from supermimic_racket_exchange import crossed_rollouts
        reference=json.loads(a.exchange_reference.read_text())
        for key,path in [('body',a.body),('calibration',a.calibration),('material',a.material),('status',status_path)]:
            if reference[key+'_sha256']!=sha(path):
                raise ValueError('Changed '+key+' for trajectory exchange')
        comparison=json.loads(a.exchange_convergence.read_text())
        if sha(a.exchange_reference) not in comparison['scale_reports_sha256'] or spec['lanes']!=[r['lane'] for r in reference['cases']]:
            raise ValueError('Different convergence cohort or case identities')
        paths={}
        initial=None
        for b in ('nominal_a','plus','minus'):
            paths[b]=[]
            measure(runs[b],body,prior,material,cfg,launches,paths=paths[b])
            r=runs[b]['raw']
            first_interval=(r['earlier_substep_traces'][spec['action_tick']] if spec['action_tick']!=spec['tick']
                            else select_substep_lanes(r['substep_trace'],spec['lanes']))
            first=first_interval['tensors']['shuttle_state_w'][0]
            if initial is None:
                initial=first
            torch.testing.assert_close(first,initial,atol=0,rtol=0)
        report=crossed_rollouts(paths,initial,prior,cfg,material,reference,comparison)
        report.update(response_sha256=sha(a.exchange_reference),convergence_sha256=sha(a.exchange_convergence),
                      contact_gradients_admitted=False,epsilon_request=spec['epsilon_request'])
        with a.output.open('x') as f:
            json.dump(report,f,indent=2,allow_nan=False)
            f.write('\n')
        return
    output = {b:measure(runs[b],body,prior,material,cfg,launches) for b in BRANCHES}
    ticks = list(range(spec['action_tick'],spec['tick']+1))
    by_lane = []
    for i,lane in enumerate(spec['lanes']):
        branch_signature = [window_signature([v['topology'][i] for v in output[b]],prior.steps) for b in BRANCHES]
        request_diff = runs['plus']['raw']['request_after'][lane] - runs['minus']['raw']['request_after'][lane]
        by_lane.append(dict(lane=lane, group=classify(branch_signature), request_difference=request_diff.tolist(),
                            signatures=dict(zip(BRANCHES,branch_signature))))
    metrics = {}
    for i,t in enumerate(ticks):
        def delta(field,left='minus',right='plus'):
            return output[right][i][field]-output[left][i][field]
        row = {}
        for predictor,target in [('measured_racket','native'),('body_racket','native'),
                                 ('body_racket_linear','native_racket_linear'),('body_racket_angular','native_racket_angular')]:
            pred,true,repeat = delta(predictor),delta(target),delta(target,'nominal_a','nominal_b')
            metrics_group = {}
            for group in ['all',*sorted({x['group'] for x in by_lane})]:
                idx = [j for j,x in enumerate(by_lane) if group=='all' or x['group']==group]
                metrics_group[group] = response_metrics(true[idx],pred[idx],repeat[idx])
            row[predictor] = metrics_group
        for j,case in enumerate(by_lane):
            case.setdefault('ticks',{})[t] = {k:delta(k)[j].tolist() for k in
                                               ('native','measured_racket','body_racket','target','final_action')}
            case['ticks'][t]['native_nominal'] = output['nominal_a'][i]['native'][j].tolist()
            case['ticks'][t]['native_left'] = delta('native','minus','nominal_a')[j].tolist()
            case['ticks'][t]['native_right'] = delta('native','nominal_a','plus')[j].tolist()
            case['ticks'][t]['native_repeat'] = delta('native','nominal_a','nominal_b')[j].tolist()
        metrics[t] = row
    report = dict(epsilon_request=spec['epsilon_request'],ticks=ticks, metrics=metrics, cases=by_lane,
                  request_directions=spec['request_directions'],prefix_sha256=run['prefix_sha256'],
                  scope='Development full-window signed differences with shuttle and spin initialized once',
                  task_contract=task,status_sha256=sha(status_path),body_sha256=sha(a.body),
                  calibration_sha256=sha(a.calibration),material_sha256=sha(a.material),
                  policy_updates=0, contact_gradients_admitted=False)
    with a.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
