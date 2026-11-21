"""Validate the TOI-only pooled proposal on recorded training starts with the actual frozen WM."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank, pool_toi_starts
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer, frozen
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import training_start_mask
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference, preview_intercept_lanes
from scripts.train_supermimic_physical_block import select_launch_for_start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','guard','archive','safety-mapping','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--intercept-plane-calibration',type=Path)
    parser.add_argument('--episodes',type=int,default=4)
    parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    meta=ck['metadata']['physical_imagination']
    paths=meta['checkpoint_paths']
    actor,adapter,template,launch,manifest=load_physical_loop(archive=args.archive,safety_mapping=args.safety_mapping,
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device=args.device)
    world_hash=tensor_digest(ck['world'].items())
    checks=[json.loads(line) for line in args.guard.read_text().splitlines()]
    if (meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256'] or
            not any(c['passed'] and c['world_sha256']==world_hash and c['world_updates']==ck['world_updates'] for c in checks)):
        raise ValueError('Different WM or missing matched forward check')
    assert ck['config']['approach_objective']=='toi_racket_only'
    adapter.load_state_dict(ck['world'])
    adapter.set_contact_bptt_mode('detached')
    actor.load_state_dict(ck['actor'])
    held=ck['config']['wm_holdout_lanes']
    ref=TaskSpaceReferenceBankV1.from_payload(ck['reference_bank'])
    ref.assert_excludes(held)
    bank=ActorStartBank(capacity=ck['actor_start_bank']['capacity'])
    bank.load_state_dict(ck['actor_start_bank'])
    cfg=Timebase(**json.loads((Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json').read_text()))
    def state(row):
        return PhysicalImaginationStateV1.from_record(dict(row,_actor_lane_mask=training_start_mask(row,held)),
                executor_template=template,launch=select_launch_for_start(row,launch),device=args.device)
    saved_plane=ck['config'].get('intercept_plane_calibration') or {}
    offset=saved_plane.get('offset_m',0.)
    if args.intercept_plane_calibration:
        from quadsmash.rl.supermimic_toi_racket_objective import calibrated_plane_offset
        from scripts.evaluate_supermimic_fresh_wm_panel import digest
        source=(saved_plane['source_checkpoint_sha256'] if digest(args.intercept_plane_calibration)==saved_plane.get('sha256')
                else digest(args.checkpoint))
        offset=calibrated_plane_offset(json.loads(args.intercept_plane_calibration.read_text()),held,source)
    trainer=PhysicalHorizonTrainer(actor,adapter,state(bank.entries[0]['start']),cfg,block_name='approach',
            reference_bank=ref,approach_objective='toi_racket_only',toi_racket_weight=ck['config']['toi_racket_weight'],toi_plane_offset_m=offset)
    trainer.critic.load_state_dict(ck['critic'])
    @torch.no_grad()
    def eligible(row):
        s=state(row)
        allowed=s.clock_goal_context['actor_lane_mask']
        match=bind_intercept_reference(s,ref,adapter.physical_model.geometry,cfg.imagination_steps*cfg.control_dt_s,plane_offset_m=offset)
        if match is None:
            return torch.zeros_like(allowed,device='cpu')
        return preview_intercept_lanes(s,match,allowed,trainer).cpu()
    row,report=bank.sample(args.attempt,eligible_lanes=eligible)
    if row is None:
        raise ValueError('No eligible TOI starts in the bank')
    rows,layout=pool_toi_starts(row,bank.same_category_starts(report['actual'],args.attempt),eligible,args.episodes,args.attempt)
    states=tuple(state(r) for r in rows)
    with frozen(adapter,trainer.critic):
        out=trainer._rollout_objective(states)
        out['loss'].backward()
        norm=sum(p.grad.square().sum() for p in actor.parameters() if p.grad is not None).sqrt()
        with torch.no_grad():
            check=trainer._rollout_objective(states,effective_weights=out['effective_weights'],baseline_toi_weights=out['toi_weights'])
        torch.testing.assert_close(out['loss'].detach(),check['loss'])
    keys=[(lane,ep) for r in layout for lane,ep,t in r['lane_episode_tick']]
    if len(keys)!=len(set(keys)) or any(lane in held for lane,ep in keys):
        raise RuntimeError('Training starts overlap or contain held lanes')
    repeats=out['tracking_nonzero_gradient_lanes']
    passed=bool(len(keys)==args.episodes and repeats==args.episodes and norm.isfinite() and norm>0 and check['toi_coverage_preserved'])
    result=dict(passed=passed,toi_plane_offset_m=offset,selected=report['actual'],requested=report['requested'],requested_episodes=args.episodes,
        layout=layout,independent_lanes_with_gradient=repeats,nonzero_approach_gradient_norm=float(norm),
        checked_objective=float(check['loss']),toi_coverage_preserved=check['toi_coverage_preserved'],
        actor_sha256=tensor_digest(actor.state_dict().items()),world_sha256=world_hash,world_updates=ck['world_updates'],
        device=args.device,actor_updates=0,critic_updates=0)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
