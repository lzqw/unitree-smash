#!/usr/bin/env python3
"""Same-window frozen checkpoint comparison on an existing native panel."""
import argparse
import json
import math
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import assess,check_panel,digest
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,tensor_digest
from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_wm_candidate_gate_v1 import compare_candidate_to_anchor


def body_forward_check(metrics, horizons, names):
    comparisons = []
    for h in horizons:
        for name in names:
            anchor = metrics['anchor'].get(h, {}).get('uniform', {})
            candidate = metrics['candidate'].get(h, {}).get('uniform', {})
            before, after = anchor.get(name), candidate.get(name)
            passed = (isinstance(before,(int,float)) and isinstance(after,(int,float))
                      and math.isfinite(before) and math.isfinite(after)
                      and candidate.get('windows',0) > 0 and candidate['windows'] == anchor.get('windows')
                      and after <= before)
            comparisons.append(dict(horizon=h,metric=name,anchor=before,candidate=after,passed=passed))
    return dict(passed=bool(comparisons) and all(row['passed'] for row in comparisons), comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','capture','candidate','anchor','archive','safety-mapping','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--body-candidate',type=Path)
    p.add_argument('--ball-candidate',type=Path,help='Training-lane gate calibration for the declared body')
    p.add_argument('--contact-calibration',type=Path,help='Frozen native racket contact fit for diagnostic transfer')
    p.add_argument('--initial-physical-components',action='store_true',
                   help='Use the ball and contact components loaded for a new policy-training process')
    p.add_argument('--teacher-racket',action='store_true',help='Use measured robot endpoints during contact transfer')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(3)
    root=Path(__file__).resolve().parents[1]
    panel=json.loads(a.panel.read_text()); measured=json.loads((a.capture/'metrics.json').read_text())
    check_panel(panel,root/'configs/g07_full_racket_eval_v4.json',root,[a.anchor,a.candidate])
    prospective = panel.get('prospective_body_comparison')
    if a.teacher_racket and not a.contact_calibration:
        p.error('Teacher racket evaluation requires analytic contact calibration')
    variant = ('calibrated_analytic' if a.contact_calibration else
               ('initial_components' if a.initial_physical_components else 'checkpoint_components'))
    if prospective and (variant not in prospective['transitions'] or a.body_candidate is None or
            str(a.body_candidate.resolve()) not in prospective['bodies'] or
            (a.contact_calibration is not None and digest(a.contact_calibration) != prospective['calibration_sha256'])):
        raise ValueError('Requested comparison was not declared before native collection')
    if a.ball_candidate:
        if not a.initial_physical_components or a.contact_calibration or not a.body_candidate:
            p.error('Ball gate calibration requires a specified body and the executable initial component transition')
        if prospective and prospective.get('ball_candidates',{}).get(str(a.ball_candidate.resolve())) != digest(a.ball_candidate):
            raise ValueError('Ball candidate was not declared before capture')
    path=a.capture/'native.pt'
    if (measured['panel_sha256']!=digest(a.panel) or measured['data_sha256']!=digest(path)
            or measured['weights']!=panel['weights'] or measured['ticks']!=800):
        raise ValueError('Native capture not bound to the complete fixed panel')
    ck=torch.load(a.candidate,mmap=True,weights_only=False,map_location='cpu')
    meta=ck['metadata']['physical_imagination']; paths=meta['checkpoint_paths']
    actor,adapter,_,_,manifest=load_physical_loop(archive=a.archive,physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        execution_root=meta['g07_execution_root'],safety_mapping=a.safety_mapping,device=a.device)
    actor.load_state_dict(ck['actor'])
    if not a.initial_physical_components:
        adapter.load_state_dict(ck['world'])
    adapter.eval()
    if (meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256'] or
            measured['actor_sha256']!=tensor_digest(actor.state_dict().items())):
        raise ValueError('Panel actor or model checkpoint identity changed')
    anchor=PhysicalWorldV1().to(a.device).eval()
    anchor.load_state_dict(torch.load(a.anchor,weights_only=True,map_location='cpu')['model'])
    if a.body_candidate:
        candidate=torch.load(a.body_candidate,map_location='cpu',weights_only=True)
        if (candidate['source_manifest']['task_contract']!=manifest['task_contract'] or
                (digest(a.body_candidate)!=meta['checkpoint_sha256']['physical'] and
                 candidate.get('calibration',{}).get('source_sha256')!=meta['checkpoint_sha256']['physical'])):
            raise ValueError('Body candidate has different robot ancestry or physics contract')
        adapter.physical_model.load_state_dict(candidate['model'],strict=True)
    if a.ball_candidate:
        new_ball=torch.load(a.ball_candidate,map_location='cpu',weights_only=True)
        from quadsmash.rl.supermimic_ball_residual_v2 import verify_event_calibration
        verify_event_calibration(new_ball,meta['checkpoint_sha256']['ball'],digest(a.body_candidate),candidate)
        adapter.ball_model.load_state_dict(new_ball['model'],strict=True)
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if a.teacher_racket and not a.contact_calibration:
        p.error('--teacher-racket requires the contact solver')
    contact_step = None
    if a.contact_calibration:
        from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, racket, tick
        settings=json.loads(a.contact_calibration.read_text())
        prior=BallImpulsePriorV1().to(a.device).eval()
        if abs(prior.dt-manifest['task_contract']['control_dt_s'])>1e-10:
            raise ValueError('Contact solver control period mismatch')
        body=adapter.physical_model
        def contact_step(inputs,observed_next,previous_patch):
            state,_,_,_,present,elapsed,launch=inputs
            prediction,logits,active=body(*inputs)
            endpoint=observed_next if a.teacher_racket else prediction
            # Recorded robot supervision intervenes only in the teacher branch. Ball always rolls forward.
            ball=body.prior(state,present,elapsed,launch)[0][:,49:].clone()
            patch=torch.zeros_like(previous_patch)
            if present.any():
                mask=present.bool()
                held_patch=previous_patch[mask]
                ball[mask],_=tick(prior,state[mask,49:],racket(state[mask],body.geometry),
                                  racket(endpoint[mask],body.geometry),settings['fitted'],
                                  settings['solver']['pair_contact_offset_m'],held_patch)
                patch[mask]=held_patch
            return torch.cat((endpoint[:,:49],ball),-1),logits.sigmoid(),active,patch
    result=assess(data,panel['cases'],anchor,adapter,panel['horizons'],a.device,contact_step=contact_step)
    if result['windows']!=measured['windows']:raise RuntimeError('Comparison window IDs changed')
    def gate_metrics(model):
        return dict(event_aligned={str(offset):{str(h):groups[f'contact{offset:+d}']
            for h,groups in model.items() if f'contact{offset:+d}' in groups} for offset in (-1,0,1)})
    result.update(contact_gate=compare_candidate_to_anchor(gate_metrics(result['metrics']['anchor']),
                  gate_metrics(result['metrics']['candidate'])),panel_sha256=measured['panel_sha256'],
                  data_sha256=measured['data_sha256'],weights=panel['weights'],
                  body_candidate_sha256=digest(a.body_candidate) if a.body_candidate else None,
                  ball_candidate_sha256=digest(a.ball_candidate) if a.ball_candidate else None,
                  contact_calibration_sha256=digest(a.contact_calibration) if a.contact_calibration else None,
                  transition=variant, initial_physical_components=a.initial_physical_components,
                  runtime_world_sha256=tensor_digest(adapter.state_dict().items()),
                  contact_input='native_robot_teacher' if a.teacher_racket else 'autoregressive_robot',
                  patch_initialization='fresh at window start' if a.contact_calibration else None,
                  actor_updates=0,formal_promotion=False)
    if prospective:
        if prospective['uniform_no_regression'] != ['body_rmse','root_angular_mae']:
            raise ValueError('Unexpected prospective body bounds')
        body_gate=body_forward_check(result['metrics'],panel['horizons'],prospective['uniform_no_regression'])
        result.update(body_forward_check=body_gate,
                      prospective_forward_passed=body_gate['passed'] and result['contact_gate']['passed'] and not a.teacher_racket)
    with a.output.open('x') as f:json.dump(result,f,allow_nan=False,indent=2)


if __name__=='__main__':
    main()
