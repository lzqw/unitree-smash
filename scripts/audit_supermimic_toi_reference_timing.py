"""Measure the error from taking successful-racket targets at the end of a contact tick."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
from quadsmash.rl.supermimic_toi_racket_objective import racket_features
from scripts.audit_supermimic_native_window_response import complete_window
from scripts.evaluate_supermimic_contact_rollout import sha
from quadsmash.rl.supermimic_impact_timing_v1 import hermite_position_velocity


def reference_label(tensors, frame, lane):
    q = tensors['racket_quaternion_w'][frame, lane]
    axis = q.new_tensor([1., 0., 0.])
    cross = torch.linalg.cross(q[1:], axis)
    normal = axis + 2 * (q[0] * cross + torch.linalg.cross(q[1:], cross))
    relative = tensors['racket_position_w'][frame, lane] - tensors['shuttle_state_w'][frame, lane, :3]
    return torch.cat((relative, tensors['racket_linear_velocity_w'][frame, lane], normal,
                      tensors['racket_angular_velocity_w'][frame, lane]))


def audit(root, status_hash):
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text())
    if sha(status_path) != status_hash or status['status'] != 'COLLECTION_FINISHED':
        raise ValueError('Expected the registered completed capture')
    source, = [r for r in status['completed'] if r['branch'] == 'nominal_a']
    folder = root / source['spec']
    if source['exit_code'] or sha(folder / 'nominal_a/native_probe.pt') != source['probe_sha256']:
        raise ValueError('Nominal trace changed')
    run = _load_branch(folder, 'nominal_a')
    spec, raw = run['spec'], run['raw']
    steps = int(raw['substep_trace']['count']) - 1
    if steps != 160:
        raise ValueError('Expected 160 substeps per control interval')
    traces = complete_window(raw, spec, steps)
    flags = raw['task_after']
    geometry = FullBodyGeometryV1(run['task_contract']['asset']['sha256'])
    cases = []
    excluded = []
    for row, lane in enumerate(spec['lanes']):
        if (raw['start']['before']['task']['had_contact'][lane] or not flags['legal_initial_contact'][lane]
                or not flags['airborne_string_hit'][lane] or flags['frame_hit'][lane] or flags['contact_count'][lane] != 1):
            excluded.append(lane)
            continue
        time = flags['contact_time'][lane]
        hits = [(j, int(i)+1) for j, tr in enumerate(traces)
                for i in torch.where(tr['tensors']['ball_clock'][1:, row] == time)[0]]
        if len(hits) != 1:
            raise ValueError('Native first-contact clock must select exactly one saved physics frame')
        j, frame = hits[0]
        t = traces[j]['tensors']
        exact = reference_label(t, frame, row)
        endpoint = reference_label(t, -1, row)
        state = raw['step_sequence'][j]['next_state'][lane:lane+1]
        fk = racket_features(state, geometry)[0]
        fk[:3] -= state[0, 49:52]
        fraction = frame / steps
        center, _ = hermite_position_velocity(t['racket_position_w'][0,row:row+1], t['racket_linear_velocity_w'][0,row:row+1],
            t['racket_position_w'][-1,row:row+1], t['racket_linear_velocity_w'][-1,row:row+1],
            state.new_tensor([frame*traces[j]['physics_dt_s']]), steps*traces[j]['physics_dt_s'])
        omega = torch.lerp(t['racket_angular_velocity_w'][0,row],t['racket_angular_velocity_w'][-1,row],fraction)
        reconstructed = torch.cat((center[0]-t['shuttle_state_w'][frame,row,:3],exact[3:9],omega))
        error = fk - exact
        cases.append(dict(lane=lane, control_tick=spec['action_tick']+j, substep=frame, native_time_s=float(time),
                          time_to_tick_end_ms=(steps-frame)*traces[j]['physics_dt_s']*1000,
                          normal_force_n=float(t['racket_force_w'][frame,row].norm()),
                          exact=exact.tolist(), endpoint=endpoint.tolist(), fk_endpoint=fk.tolist(),
                          reconstructed_center_error_mm=1000*float((reconstructed-exact)[:3].norm()),
                          reconstructed_angular_velocity_error_rad_s=float((omega-exact[9:]).norm()),
                          native_relative_position_error_mm=1000*float((endpoint-exact)[:3].norm()),
                          bank_relative_position_error_mm=1000*float(error[:3].norm()),
                          bank_normal_projection_error_mm=1000*float(error[:3] @ exact[6:9]),
                          bank_velocity_error_mps=float(error[3:6].norm()),
                          bank_angular_velocity_error_rad_s=float(error[9:12].norm())))
    metrics = {}
    for key in ('time_to_tick_end_ms','native_relative_position_error_mm','bank_relative_position_error_mm',
                'bank_velocity_error_mps','bank_angular_velocity_error_rad_s','reconstructed_center_error_mm','reconstructed_angular_velocity_error_rad_s'):
        values = torch.tensor([r[key] for r in cases])
        metrics[key] = dict(median=float(values.quantile(.5)),p90=float(values.quantile(.9))) if len(values) else None
    return dict(status_sha256=status_hash, branch_sha256=source['probe_sha256'], selected=len(spec['lanes']),
                role=spec['data_role'], excluded=excluded, cases=cases, metrics=metrics)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training',type=Path,nargs='+',required=True,help='Scored training directories with support.json and raw/')
    cli.add_argument('--development',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    torch.set_num_threads(2)
    groups={}
    for folder in args.training:
        support=json.loads((folder/'support.json').read_text())
        if not support['training_only'] or support['policy_training_admitted'] or support['contact_receipt']:
            raise ValueError('Expected a training-only intervention source')
        groups[folder.name]=audit(folder/'raw',support['status_sha256'])
    src=json.loads((args.development/'completed.json').read_text())
    groups['development_wide']=audit(args.development/'wide',src['wide'])
    if len({g['status_sha256'] for g in groups.values()}) != 1+len(args.training):
        raise ValueError('Repeated input panel')
    result=dict(groups=groups,actor_updates=0,contact_receipt=False,
                scope='One nominal branch per launch. Single legal airborne string contacts in the observed window. Episode-end success was not measured by these probes.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
