#!/usr/bin/env python3
"""Check native action-to-racket transfer after fitting a pose readout; use the same fixed contact solver."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_toi_slip import TOISlip, body_transfer
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_native_racket_toi_head import g07
from quadsmash.control.reactive_landing_v1 import quat_matrix
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def channels(frame):
    position, quaternion, linear, angular = frame
    return dict(center=position,normal=quat_matrix(quaternion)[:,:,0],linear=linear,angular=angular)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('body','replay','stage','candidate','collection','slip-fit','calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--attribute-drift', action='store_true',
                        help='Re-anchor only the 49 robot coordinates at recorded control boundaries')
    a=parser.parse_args()
    torch.set_num_threads(2)
    config=json.loads(a.calibration.read_text())
    prior=BallImpulsePriorV1(normal_impulse_scale=config['fitted']['normal_impulse_scale']).eval()
    reference=json.loads(a.slip_fit.read_text())
    if (reference['g07_sha256']!=sha(a.collection/'status.json') or reference['calibration_sha256']!=sha(a.calibration)
            or reference['final_holdout_opened']):
        raise ValueError('Contact calibration or comparison dataset changed')
    fit=json.loads((a.candidate/'fit.json').read_text())
    candidate=a.candidate/'world.pt'
    previous=json.loads((a.replay/'fit.json').read_text())
    if (sha(candidate)!=fit['model_sha256'] or fit['source_body_sha256']!=sha(a.body) or
            fit['source_world_sha256']!=previous['online_world_sha256'] or fit['source_stage_sha256']!=sha(a.stage)):
        raise ValueError('Candidate does not follow the repaired Body baseline')
    slip=TOISlip(reference['selected']['friction'],reference['selected']['sticking'])
    _,_,repeat,groups,seq,commands=g07(a.collection,prior)
    measured={branch:[channels([f[-1] for f in stream]) for _,_,stream in path] for branch,path in seq.items()}
    folder=a.collection/json.loads((a.collection/'status.json').read_text())['completed'][0]['spec']
    keys=('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
    for b in ('nominal_a','nominal_b'):
        run=_load_branch(folder,b)
        measured[b]=[channels([tr['tensors'][k][-1] for k in keys]) for tr in complete_window(run['raw'],run['spec'],prior.steps)]
    if set(commands['lanes']) & {i for ids in fit['training']['included_lanes'].values() for i in ids}:
        raise ValueError('Pose fit includes a development probe lane')
    stable=torch.tensor([g=='racket_stable' for g in groups])
    results={}
    for stage,path,replay,source in [('repaired',a.body,a.replay,a.stage),('center_fit',candidate,None,None)]:
        outputs={}
        ball_result=body_transfer(a.collection,path,prior,slip,groups,repeat,seq,commands,replay,source,racket_outputs=outputs)
        timeline={}
        for h in (1,2,4,8,16):
            pred={b:channels(r[h-1]) for b,r in outputs.items()}
            native={b:r[h-1] for b,r in measured.items()}
            timeline[h]={}
            for key in pred['plus']:
                true=native['plus'][key]-native['minus'][key]
                change=pred['plus'][key]-pred['minus'][key]
                noise=native['nominal_b'][key]-native['nominal_a'][key]
                point=torch.cat([pred[b][key]-native[b][key] for b in pred])
                timeline[h][key]=dict(point_rmse=float(point.square().mean().sqrt()),
                    all_pairs=response_metrics(true,change,noise),
                    stable_pairs=response_metrics(true[stable],change[stable],noise[stable]) if stable.any() else None)
        results[stage]=dict(racket=timeline,ball=ball_result['metrics'])
    expected=reference['online_body_racket']['metrics']['absolute_rmse']
    if abs(expected-results['repaired']['ball']['absolute_rmse'])>1e-6:
        raise ValueError('Repaired baseline does not reproduce on the comparison')
    evidence=dict(scores=results,lane_ids=commands['lanes'],source_slip_sha256=sha(a.slip_fit),fit_sha256=sha(a.candidate/'fit.json'),
                  actor_updates=0,final_holdout_opened=False,contact_gradients_admitted=False)
    if a.attribute_drift:
        evidence['control_boundary_oracles'] = {}
        for interval in (1, 4, 8):
            anchored = body_transfer(a.collection, a.body, prior, slip, groups, repeat, seq, commands,
                                     a.replay, a.stage, body_anchor_interval=interval)
            evidence['control_boundary_oracles'][interval] = dict(
                metrics=anchored['metrics'], first_racket_impact_time_s=anchored['first_racket_impact_time_s'],
                predicted_delta_mps=anchored['predicted_delta_mps'])
    with a.output.open('x') as f:
        json.dump(evidence,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
