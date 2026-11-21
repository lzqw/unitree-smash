#!/usr/bin/env python3
"""Measure reference coverage and actor gradients on saved causal Recovery starts."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_approach_data_v1 import gradient_parts
from quadsmash.rl.supermimic_horizon_learner_v2 import frozen
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES, load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import RECOVERY_TICKS, load_start_cache, training_start_mask
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from scripts.evaluate_supermimic_fresh_wm_panel import digest
from scripts.train_supermimic_physical_block import select_launch_for_start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','archive','safety-mapping','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--wm-forward-guard',type=Path)
    parser.add_argument('--resume-wm-replay',type=Path)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
    parser.add_argument('--effective-start-lanes',type=int,default=1)
    parser.add_argument('--pooled-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.effective_start_lanes<=128 or (args.pooled_only and args.effective_start_lanes==1):
        parser.error('A pooled audit needs a target of 2 to 128 lanes')
    if bool(args.wm_forward_guard)!=bool(args.resume_wm_replay):
        parser.error('A candidate audit requires both the preregistered WM replay and the native guard')
    torch.set_num_threads(3)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=ck['config']
    wm=ck['metadata']['physical_imagination']
    paths=wm['checkpoint_paths']
    actor,adapter,template,launch,manifest=load_physical_loop(
        archive=args.archive,safety_mapping=args.safety_mapping,physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        execution_root=wm['g07_execution_root'],device=args.device)
    if adapter.manifest['checkpoint_sha256']!=wm['checkpoint_sha256']:
        raise ValueError('Different checkpoint dependencies')
    held=cfg['wm_holdout_lanes']
    world=ck['world']
    handoff=None
    guard=None
    if args.resume_wm_replay:
        from scripts.supermimic_online_forward_guard import OnlineForwardGuard
        guard=OnlineForwardGuard(args.wm_forward_guard,adapter,manifest['task_contract'])
        fit=json.loads((args.resume_wm_replay/'fit.json').read_text())
        recipe={k:fit[k] for k in ('separate_supervision','natural_flight_batch','natural_flight_loss','active_flight_batch','direct_long_body_supervision')}
        world,_,handoff=guard.replay_for_resume(args.resume_wm_replay,args.checkpoint,ck,held,recipe)
    actor.load_state_dict(ck['actor'],strict=True)
    adapter.load_state_dict(world,strict=True)
    adapter.set_contact_bptt_mode('detached')
    online_check=guard.check(adapter) if guard is not None else None
    if online_check is not None and not online_check['passed']:
        raise ValueError('Actual loaded WM failed its native forward check')
    bank=TaskSpaceReferenceBankV1.from_payload(ck['reference_bank']).to(args.device)
    bank.assert_excludes(held)
    if not bank.manifest['protocol_verified'] or bank.geometry_sha256!=adapter.physical_model.geometry.model_sha256:
        raise ValueError('Recovery reference bank geometry or scoring is unverified')
    starts=load_start_cache(args.archive,manifest)
    pools={k:ck['dynamic_stage_starts'][k] for k in ('touchdown','post_touchdown','recovery_failure','stable_recovery')}
    pools['initial_archive']=[starts[tick] for tick in RECOVERY_TICKS]
    tb=Timebase(**json.loads((Path(__file__).resolve().parents[1]/'configs/supermimic_task_horizon_v2.json').read_text()))
    results={}
    families={}
    trainer=None
    for name,rows in pools.items():
        masks=[training_start_mask(row,held,recovery=True) for row in rows]
        eligible=[i for i,mask in enumerate(masks) if mask.any()]
        families[name]=[(rows[i],masks[i]) for i in eligible]
        result=dict(starts=len(rows),valid_starts=len(eligible),selected_lanes_by_start=[int(m.sum()) for m in masks])
        results[name]=result
        if not eligible:
            continue
        i=eligible[0]
        row=dict(rows[i],_actor_lane_mask=masks[i])
        state=PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=select_launch_for_start(row,launch),device=args.device)
        if trainer is None:
            trainer=PhysicalHorizonTrainer(actor,adapter,state,tb,block_name='recovery',trainable_names=RECOVERY_NAMES,
                                          reference_bank=bank,tracking_weight=cfg['tracking_weight'])
            trainer.critic.load_state_dict(ck['critic'],strict=True)
        if args.pooled_only:
            continue
        with frozen(adapter,trainer.critic):
            rollout=trainer._rollout_objective(state)
            grads=gradient_parts({k:rollout[k] for k in ('task','anchor','tracking')},[p for p in actor.parameters() if p.requires_grad])
            result.update(first_verified_start=i,coverage=rollout['reference_coverage'],gradients=grads,
                          lane_steps=rollout['tracking_active_lane_steps'],nonzero_lanes=rollout['tracking_nonzero_gradient_lanes'])
            del rollout
    if trainer is None:
        raise ValueError('No eligible Recovery start')
    pooled=None
    if args.effective_start_lanes>1:
        from quadsmash.rl.supermimic_recovery_batches import recovery_batch
        rows,layout=recovery_batch(families,args.effective_start_lanes,1)
        states=tuple(PhysicalImaginationStateV1.from_record(row,executor_template=template,
                         launch=select_launch_for_start(row,launch),device=args.device) for row in rows)
        with frozen(adapter,trainer.critic):
            result=trainer._rollout_objective(states)
            grads=gradient_parts({k:result[k] for k in ('task','anchor','tracking')},[p for p in actor.parameters() if p.requires_grad])
            with torch.no_grad():
                repeat=trainer._rollout_objective(states,effective_weights=result['effective_weights'])
                torch.testing.assert_close(result['loss'].detach(),repeat['loss'],rtol=1e-5,atol=1e-6)
            pooled=dict(requested=args.effective_start_lanes,layout=layout,gate_lanes=result['gate_lanes'],
                        coverage=result['reference_coverage'],gradients=grads,valid_critic_lane_steps=int(result['critic_data'][-1].sum()))
    hashes=[(actor,ck['actor']),(adapter,world),(trainer.critic,ck['critic'])]
    if any(tensor_digest(module.state_dict().items())!=tensor_digest(saved.items()) for module,saved in hashes):
        raise RuntimeError('Read-only audit changed model weights')
    report=dict(checkpoint_sha256=digest(args.checkpoint),actor_sha256=tensor_digest(ck['actor'].items()),
                world_sha256=tensor_digest(world.items()),optimizer_updates=0,pools=results,pooled=pooled,
                wm_handoff=handoff,forward_check=online_check,scope='Offline gradient audit on recorded, causally gated Recovery starts')
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
