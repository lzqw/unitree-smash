#!/usr/bin/env python3
"""Run an eight-environment native prefix and the actual TOI Approach loss, without stepping an optimizer."""
import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_evidence_binding_v1 import json_sha
    from quadsmash.rl.supermimic_horizon_learner_v2 import frozen
    from quadsmash.rl.supermimic_physical_capture_v1 import capture_controller, capture_state
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES, freeze_execution_modules, load_physical_loop, tensor_digest
    from quadsmash.rl.supermimic_policy_execution_v1 import contract as execution_contract, policy_forward
    from quadsmash.rl.supermimic_run_lifecycle import close_resource, stage
    from quadsmash.rl.supermimic_task_contract import make_task_cfg, require_resume_contract, resolve_task_contract, zero_panel_reset_noise
    from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
    from quadsmash.rl.supermimic_timebase_v2 import Timebase
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1

    class ShadowEnv(SuperMimicEnvV1, FrozenPanelEnv):
        pass

    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    ck=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    meta=ck['metadata']; identity=meta['physical_imagination']
    paths=identity['checkpoint_paths']
    actor,adapter,_,launch,manifest=load_physical_loop(archive=args.archive, physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],action_ball_checkpoint=paths.get('action_ball'),
        safety_mapping=args.safety_mapping,execution_root=identity['g07_execution_root'],device=args.device)
    if identity['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256']:
        raise ValueError('Shadow checkpoint components differ from native training')
    if meta['policy_execution']!=execution_contract():
        raise ValueError('Shadow requires the training causal policy execution')
    guard=None
    world=ck['world']
    handoff=None
    expected_guard=ck['config'].get('wm_forward_guard_sha256')
    if args.wm_forward_guard:
        from scripts.supermimic_online_forward_guard import OnlineForwardGuard
        guard=OnlineForwardGuard(args.wm_forward_guard,adapter,manifest['task_contract'])
    if expected_guard is not None and (guard is None or guard.spec_sha256!=expected_guard):
        raise ValueError('TOI validation must retain the checkpoint-bound forward panel')
    if args.resume_wm_replay:
        # Keep the training recipe, optimizer and anchor sampling clock from the scored WM.
        recipe = {}
        if args.canary_updates:
            fit = json.loads((args.resume_wm_replay/'fit.json').read_text())
            recipe = {key: fit[key] for key in ('natural_flight_batch', 'natural_flight_loss',
                      'active_flight_batch', 'direct_long_body_supervision', 'separate_supervision', 'learning_rate')}
        world, wm_state, handoff = guard.replay_for_resume(
            args.resume_wm_replay, args.checkpoint, ck, ck['config']['wm_holdout_lanes'], recipe)
        if args.canary_updates:
            ck = dict(ck, world_optimizer=wm_state['optimizer'], natural_flight_rng=wm_state['natural_flight_rng'],
                      config={**ck['config'], **{'wm_'+key: value for key, value in recipe.items()}},
                      wm_replay_handoff=handoff)
    actor.load_state_dict(ck['actor'],strict=True)
    adapter.load_state_dict(world,strict=True)
    if guard is not None:
        check=guard.check(adapter)
        (args.output/'initial_forward_check.json').write_text(json.dumps(check,indent=2,allow_nan=False))
        if not check['passed']:raise ValueError('TOI shadow WM failed native forward validation')
    actor.eval();adapter.eval();adapter.set_contact_bptt_mode('detached')
    actor_sha=tensor_digest(actor.state_dict().items());world_sha=tensor_digest(adapter.state_dict().items())
    timebase=Timebase(**json.loads((root/'configs/supermimic_task_horizon_v2.json').read_text()))
    bank=TaskSpaceReferenceBankV1.from_payload(torch.load(args.reference_bank,map_location='cpu',weights_only=True))
    if (not bank.manifest['protocol_verified'] or bank.geometry_sha256!=adapter.physical_model.geometry.model_sha256
            or bank.execution_contract_sha256!=json_sha(meta['policy_execution'])):
        raise ValueError('Reference physics or scoring contract mismatch')
    if args.canary_updates:
        eligible = [lane for lane in range(128) if lane not in bank.excluded_lanes]
        lanes = eligible[:32]
        if len(lanes) != 32:
            raise ValueError('Fewer than 32 eligible cases outside the reserved panel')
    else:
        lanes=list(range(8))
        bank.assert_excludes(lanes)
    bank=bank.to(args.device)
    source_path=Path(paths['source'])
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    task,protocol,cases,receipt=resolve_task_contract(source,source_path,execution_root=Path(identity['g07_execution_root']),root=Path(identity['g07_execution_root']))
    require_resume_contract(ck,task)
    if task!=manifest['task_contract']:
        raise ValueError('Archived and native task definitions differ')
    cfg=make_task_cfg(len(lanes),task,protocol,receipt);cfg.seed=args.seed;cfg.sim.device=args.device
    env=None;failed=True
    try:
        env=ShadowEnv(cfg,cases=[cases[i] for i in lanes],reset_noise=zero_panel_reset_noise(len(lanes),args.seed),
            residual_policy_checkpoint=source,g07_execution_root=Path(identity['g07_execution_root']),
            native_return_gamma=timebase.gamma,contact_flight_recovery=task['contact_flight_recovery'])
        freeze_execution_modules(env._sm_residual_controller)
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        obs,_=wrapper.reset()
        stage(args.output,'NATIVE_RESET',num_envs=env.num_envs,case_ids=[cases[i]['case_id'] for i in lanes])
        budget_rows = []
        with torch.no_grad():
            for tick in range(args.tick):
                if args.canary_updates and tick in (224, 240, 248):
                    budget_rows.append(dict(before=capture_state(env), executor_before=capture_controller(env._sm_residual_controller),
                                            observation=obs['policy'].detach().clone()))
                request=policy_forward(actor,obs['policy'],env.rally,sample=False)[0]
                obs,_,done,_=wrapper.step(request)
                if done.any():raise RuntimeError('Shadow prefix crossed an episode boundary')
                if (tick+1)%64==0:stage(args.output,'PREFIX_PROGRESS',ticks=tick+1)
            row=dict(before=capture_state(env),executor_before=capture_controller(env._sm_residual_controller),
                     observation=obs['policy'].detach().clone())
            request=policy_forward(actor,obs['policy'],env.rally,sample=False)[0]
            obs,_,done,_=wrapper.step(request)
            if done.any():raise RuntimeError('Command parity step crossed an episode boundary')
            executed=env._sm_joint_targets.detach().clone()
            torch.save(dict(row=row,executed=executed,launch=launch[lanes],case_ids=[cases[i]['case_id'] for i in lanes],
                            tick=args.tick,seed=args.seed,actor_sha256=actor_sha,world_sha256=world_sha),args.output/'native_start.pt')
            state=PhysicalImaginationStateV1.from_record(row,executor_template=env._sm_residual_controller,
                                                       launch=launch[lanes],device=args.device)
        stage(args.output,'NATIVE_PREFIX_CAPTURED',tick=args.tick)
        trainer=PhysicalHorizonTrainer(actor,adapter,state,timebase,trainable_names=APPROACH_NAMES,block_name='approach',
                                      command_delta_budget=ck['config'].get('command_delta_budget',.005),
                                      target_delta_budget_rad=ck['config'].get('target_delta_budget_rad',.04),
                                      native_reference_command_budget=ck['config'].get('native_reference_command_budget'),
                                      native_reference_target_budget_rad=ck['config'].get('native_reference_target_budget_rad'),
                                      reference_bank=bank,approach_objective='toi_racket_only',toi_racket_weight=1.,
                                      toi_plane_offset_m=(ck['config'].get('intercept_plane_calibration') or {}).get('offset_m',0.))
        trainer.critic.load_state_dict(ck['critic'],strict=True)
        critic_sha=tensor_digest(trainer.critic.state_dict().items())
        with frozen(adapter,trainer.critic):
            rollout=trainer._rollout_objective(state)
            params=[p for p in actor.parameters() if p.requires_grad]
            gradients=torch.autograd.grad(rollout['tracking'],params,allow_unused=True)
            norm=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
            with torch.no_grad():
                _,_,preview_target=trainer._preview_command(state.branch())
                command_error=float((preview_target-executed).abs().max())
        unchanged=(actor_sha==tensor_digest(actor.state_dict().items()) and world_sha==tensor_digest(adapter.state_dict().items())
                   and critic_sha==tensor_digest(trainer.critic.state_dict().items()))
        passed=bool(unchanged and rollout['tracking_effective_weight_sum']>0 and 0<norm<float('inf') and command_error<=2e-6)
        result=dict(passed=passed,num_envs=env.num_envs,case_ids=[cases[i]['case_id'] for i in lanes],seed=args.seed,
                    native_prefix_ticks=args.tick,imagination_steps=timebase.imagination_steps,actor_sha256=actor_sha,
                    world_sha256=world_sha,critic_sha256=critic_sha,wm_replay_handoff=handoff,checkpoint=str(args.checkpoint),
                    reference_bank_sha256=hashlib.sha256(args.reference_bank.read_bytes()).hexdigest(),
                    matched_references=rollout['matched_references'],effective_gate_lanes=rollout['gate_lanes'],
                    tracking_active_lane_steps=rollout['tracking_active_lane_steps'],
                    tracking_weight_sum=rollout['tracking_effective_weight_sum'],toi_actor_gradient_norm=norm,
                    max_native_command_error_rad=command_error,weights_unchanged=unchanged,
                    actor_updates=0,critic_updates=0,world_updates=0,contact_gradients='detached',
                    scope='native prefix plus frozen-model BPTT; no paired native Jacobian or performance receipt')
        (args.output/'shadow.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if not passed:raise RuntimeError('TOI shadow failed; see shadow.json')
        if args.canary_updates:
            from run_supermimic_toi_native_canary import learn
            del rollout, gradients
            budget_states = [PhysicalImaginationStateV1.from_record(item, executor_template=env._sm_residual_controller,
                             launch=launch[lanes], device=args.device) for item in budget_rows]
            learn(args, budget_states=budget_states, env=env, wrapper=wrapper, observation=obs, trainer=trainer, state=state,
                  launch=launch[lanes], lanes=lanes, bank=bank, ck=ck, task=task, protocol=protocol,
                  forward_guard=guard,full_launch=launch)
        failed=False
    finally:
        close_resource(args.output,'ENV',env,primary_failed=failed)


def main(*, canary=False):
    from isaaclab.app import AppLauncher
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native
    p=argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(p)
    for arg in ('checkpoint','archive','reference-bank','safety-mapping','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--wm-forward-guard',type=Path)
    p.add_argument('--resume-wm-replay',type=Path)
    if canary:
        p.description='32 native training cases, supervised WM refresh, bounded TOI Approach updates and frozen RTG.'
        p.set_defaults(canary_updates=8)
        p.add_argument('--packed-data', type=Path, required=True)
        p.add_argument('--wm-action-collection', type=Path, required=True)
    else:
        p.set_defaults(canary_updates=0)
    p.add_argument('--tick',type=int,default=256)
    p.add_argument('--seed',type=int,default=882072)
    a=p.parse_args()
    if a.resume_wm_replay and not a.wm_forward_guard:p.error('A replay candidate requires its native forward guard')
    if not 0<a.tick<800:p.error('Select a control tick within the episode')
    a.output.mkdir(parents=True,exist_ok=False)
    app=AppLauncher(a,fast_shutdown=False).app
    execute_scoped_native(a.output,lambda:run(a),app)


if __name__=='__main__':
    from quadsmash.rl.supermimic_run_lifecycle import native_process_main
    native_process_main(main)
