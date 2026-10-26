#!/usr/bin/env python3
"""Learn bounded whole-body residuals on the original full-model TRAIN suite."""
import argparse,copy,hashlib,json,signal,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--base-checkpoint',type=Path,required=True)
p.add_argument('--body-source-evaluation',type=Path,required=True)
p.add_argument('--config',type=Path,default=ROOT/'configs/g07_full_racket_residual_ppo_v4.json')
initialization_options=p.add_mutually_exclusive_group()
initialization_options.add_argument('--resume',type=Path)
initialization_options.add_argument('--initialize-actor',type=Path,
    help='Explicitly transfer a compatible canonical public actor into a fresh joint-reward learner; no critic/Adam/physics/RNG/counter resume')
initialization_options.add_argument('--initialize-centered-policy',type=Path,
    help='Transfer an independently measured outer actor on the same fixed base into a fresh learner; no critic/Adam/physics/RNG/counter resume')
initialization_options.add_argument('--initialize-centered-action-units',type=Path,
    help='Explicit fresh measured-actor transfer to leg0.06 units with witnessed physical mean checks; not strict resume')
p.add_argument('--centered-source-evaluation',type=Path,
    help='Required original TRAIN128 native evaluation of --initialize-centered-policy')
p.add_argument('--resize-ground',action='store_true',help='Explicitly fork a canonical learner onto a larger fresh ground batch; no old physical state restoration')
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--updates',type=int,default=8)
p.add_argument('--seed',type=int,default=748501)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
centered_initialization=(args.initialize_centered_policy is not None or args.initialize_centered_action_units is not None)
if centered_initialization!=(args.centered_source_evaluation is not None):
    p.error('Supply a centered initialization option together with --centered-source-evaluation')
app=AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.full_racket_residual_env_v4 import FullRacketResidualTrainingEnvV4
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_residual_reward_v4 import EVENT_WEIGHTS
from quadsmash.control.full_racket_body_residual_v4 import baseline_mode,validate_frozen_baseline
from quadsmash.control.measured_joint_placement_reward_v4 import placement_settings,REWARD_VERSION
from quadsmash.control.measured_string_alignment_reward_v4 import alignment_settings,REWARD_VERSION as ALIGNMENT_REWARD_VERSION
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.rl.episode_trace_v4 import EpisodeTraceRecorderV4,trace_settings,SCHEMA as TRACE_SCHEMA
from quadsmash.rl.full_racket_body_residual_ppo_v4 import (
    ResidualRslAdapterV4,ResidualTrainingInterrupted,export_residual_policy,save_completed_update_error_checkpoint)
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.usd_dependencies_v4 import audit_composed_asset

stopping=False
def stop(*_):
    global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def atomic_torch(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def joined_episodes(env,cases,indices,protocol):
    assert len(env.rally_records)==len(env.episode_records),'Native rally/jump episode count mismatch'
    rows=[];checked=[]
    for rally,jump in zip(env.rally_records,env.episode_records):
        assert rally['env_id']==jump['env_id'],'Native rally/jump episode order mismatch'
        lane=rally['env_id'];index=indices[lane%len(indices)]
        row=dict(rally,jump=jump,case_id=f'TRAIN{index}_LANE{lane}_EP{rally["episode_index"]}',
                 source_training_index=index,ground_init=True)
        rows.append(row);checked.append(independent_episode(row,protocol))
    return rows,dict(completed_episodes=len(rows),air=sum(v['airborne_contact'] for v in checked),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in checked),
        complete=sum(v['complete'] for v in checked),numerical_failures=sum(r['numerical_failures'] for r in rows))


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'INPUTS.json').exists():raise ValueError('Use a new native output directory')
    if args.updates<1:raise ValueError('At least one actual PPO update required')
    if any(x is not None for x in [args.initialize_actor,args.initialize_centered_policy,args.initialize_centered_action_units]) or args.resize_ground:
        raise ValueError('Explicit fresh zero body residual or its own canonical resume only')
    if args.resize_ground and args.resume is None:raise ValueError('--resize-ground requires an actual canonical --resume checkpoint')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol,None)
    if args.num_envs<len(cases) or args.num_envs%len(cases):
        raise ValueError('Use a positive multiple of all128 fixed TRAIN cases')
    configuration=json.loads(args.config.read_text());scales=configuration['joint_scale_rad']
    if scales != [.15]*18 or configuration.get('fresh_exploration_std') != [.2]*12+[.1]*6:
        raise ValueError('Require the explicit larger physical exploration configuration')
    mode=baseline_mode(configuration.get('base_controller_mode'))
    if centered_initialization and mode!='frozen_public_policy':
        raise ValueError('Centered initialization requires the same fixed learned baseline')
    base=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False)
    source_verification=None
    if mode=='analytic':
        if not base.get('native_execution') or base['schema']!='full-racket-ball-cem-v4':
            raise ValueError('Require one actually measured baseline controller, no inherited residual actor')
    else:
        if args.initialize_actor is not None:
            raise ValueError('A centered residual starts fresh; do not add the old actor twice')
        validate_frozen_baseline(base)
        from quadsmash.rl.frozen_body_source_v4 import verify_frozen_body_source
        source_verification=verify_frozen_body_source(args.base_checkpoint,base,args.body_source_evaluation,ROOT,digest(protocol_path))
    if base['source_public_policy']['seed'].get('arm_target_rate_limit_rad_s')!=5.:
        raise ValueError('Use a measured baseline with the same5rad/s arm target limiter')
    if base['asset_sha256']!=protocol['asset']['sha256'] or digest(ROOT/protocol['asset']['path'])!=protocol['asset']['sha256']:
        raise ValueError('Baseline/full-model asset mismatch')
    for name,expected in protocol['physical_configs'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Physical config changed: '+name)
    joint_placement=placement_settings(configuration.get('joint_placement_reward'))
    live_alignment=alignment_settings(configuration.get('live_alignment_reward'))
    episode_trace=trace_settings(configuration.get('episode_trace'))
    if live_alignment is not None:
        if joint_placement is None:
            raise ValueError('Measured strings/attack shaping requires explicit joint placement')
        if live_alignment['discount']!=configuration['runner']['algorithm']['gamma']:
            raise ValueError('Potential discount must equal the actual learner gamma')
    if args.initialize_actor is not None and joint_placement is None:
        raise ValueError('--initialize-actor requires the explicit joint-placement reward config')
    contract=dict(base_policy_sha256=digest(args.base_checkpoint),protocol_sha256=digest(protocol_path),
        joint_scale_rad=scales,reward_version='native-full-chain-residual-v1',event_weights=EVENT_WEIGHTS,
        actor_warmup_updates=configuration['actor_warmup_updates'],seed=args.seed)
    if mode=='frozen_public_policy':contract['base_controller_mode']=mode
    if joint_placement is not None:
        contract.update(reward_version=REWARD_VERSION,joint_placement_reward=joint_placement)
    if live_alignment is not None:
        contract.update(reward_version=ALIGNMENT_REWARD_VERSION,live_alignment_reward=live_alignment,
            target_event_requires_actual_attack=True,fixed_upper_uv_position_shaping=False)
    contract.update(fresh_exploration_std=configuration['fresh_exploration_std'],base_controller_schema=base['schema'],fixed_source_actor_count=2)
    train_cfg=copy.deepcopy(configuration['runner']);train_cfg['seed']=args.seed;train_cfg['residual_contract']=contract
    if episode_trace is not None:train_cfg['episode_trace']=dict(schema=TRACE_SCHEMA,**episode_trace)
    if train_cfg['obs_groups']['policy']!=['policy'] or train_cfg['policy']['class_name']!='ActorCritic' or train_cfg['policy']['noise_std_type']!='log':
        raise ValueError('This adapter requires the declared public-only feedforward log-std actor')
    cfg=full_racket_cfg(args.num_envs,protocol['physics_dt_s']);cfg.seed=args.seed
    cfg.episode_length_s=protocol['episode_seconds']
    env=FullRacketResidualTrainingEnvV4(cfg,cases=cases*(args.num_envs//len(cases)),
        continue_after_miss=True,joint_placement_reward=joint_placement,live_alignment_reward=live_alignment)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    adapter=ResidualRslAdapterV4(wrapped,base,scales,lambda:stopping,base_mode=mode)
    runner=None;completed_update_safe=False
    try:
        if env.robot.is_fixed_base:raise ValueError('Actual floating ground-start robot required')
        atomic_json(args.output/'asset_dependencies.json',audit_composed_asset(ROOT,cfg.robot.spawn.usd_path))
        atomic_json(args.output/'INPUTS.json',dict(contract=contract,training_indices=indices,
            num_envs=args.num_envs,requested_updates=args.updates,runner=train_cfg,
            configuration_sha256=digest(args.config),resume_checkpoint_sha256=digest(args.resume) if args.resume else None,
            episode_trace_settings=episode_trace,
            frozen_baseline_source_verification=source_verification,
            initial_actor_checkpoint_sha256=digest(args.initialize_actor) if args.initialize_actor else None,
            initial_centered_policy_sha256=digest(args.initialize_centered_policy) if args.initialize_centered_policy else None,
            initial_centered_action_units_policy_sha256=digest(args.initialize_centered_action_units) if args.initialize_centered_action_units else None,
            centered_source_evaluation=str(args.centered_source_evaluation) if args.centered_source_evaluation else None,
            resize_ground_requested=args.resize_ground,
            physics_dt_s=cfg.sim.dt,policy_dt_s=env.step_dt,formal_cases_read=False,
            formal_acceptance=False,scope='Native TRAIN learning across changing weights; no frozen-policy performance.'))
        runner=NativeRunner(adapter,train_cfg,log_dir=str(args.output/'ppo'),device=env.device)
        initialization=dict(mode='fresh_zero_mean_actor',num_envs=args.num_envs,initial_iteration=0)
        if mode=='frozen_public_policy':
            initialization.update(mode='fresh_zero_mean_actor_over_frozen_public_base',
                fixed_base_policy_sha256=digest(args.base_checkpoint),trainable_action_joints=18)
        if args.resume:
            previous=torch.load(args.resume,map_location='cpu',weights_only=False)
            if previous.get('infos',{}).get('residual_contract')!=contract:
                raise ValueError('Residual baseline/scales/rewards changed across resume')
            if args.resize_ground:
                from scripts.residual_ground_resize_v4 import load_resized_ground
                initialization=load_resized_ground(runner,args.resume)
            else:
                runner.load(str(args.resume),load_optimizer=True)
                initialization=dict(mode='strict_canonical_ground_resume',num_envs=args.num_envs,
                                    initial_iteration=runner.current_learning_iteration)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_actor:
            from scripts.initialize_residual_actor_v4 import initialize_actor
            initialization=initialize_actor(runner,args.initialize_actor)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_centered_policy:
            from scripts.initialize_centered_policy_v4 import initialize_centered_policy
            initialization=initialize_centered_policy(runner,args.initialize_centered_policy,
                args.centered_source_evaluation,base,ROOT)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_centered_action_units:
            from scripts.initialize_centered_action_units_v4 import initialize_centered_action_units
            initialization=initialize_centered_action_units(runner,args.initialize_centered_action_units,
                args.centered_source_evaluation,base,ROOT)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        else:
            from scripts.initialize_fresh_body_residual_v4 import initialize_fresh_body_actor
            initialization.update(initialize_fresh_body_actor(runner.alg.policy,configuration))
            if runner.alg.optimizer.state:raise ValueError('Fresh zero actor must keep empty Adam state')
        if not args.resume:
            atomic_torch(args.output/'fresh_initial_model_state.pt',dict(
                model_state_dict={name:value.detach().cpu().clone() for name,value in runner.alg.policy.state_dict().items()},
                base_policy_sha256=digest(args.base_checkpoint),actual_control_env_steps=adapter.total_control_env_steps,
                optimizer_entries=len(runner.alg.optimizer.state),fixed_source_residual_actor_count=2))
        input_record=json.loads((args.output/'INPUTS.json').read_text())
        input_record['initialization']=initialization
        atomic_json(args.output/'INPUTS.json',input_record)
        data=env.robot.data;ground_root=data.root_state_w.detach().cpu().clone()
        ground_root[:,:3]-=env.scene.env_origins.cpu()
        expected_root=torch.tensor(protocol['initialization']['root_position_m'],dtype=ground_root.dtype)
        if not torch.allclose(ground_root[:,:3],expected_root[None].expand(args.num_envs,-1),atol=1e-5,rtol=0):
            raise ValueError('Fresh native ground positions differ from the original protocol')
        if env.episode_length_buf.any():raise ValueError('Training must begin at actual episode clock zero')
        atomic_json(args.output/'initial_ground_states.json',dict(scope='Actual native state before this job steps; no physical continuation claim across batch resizing.',
            initialization=initialization,source_training_indices=[indices[i%len(indices)] for i in range(args.num_envs)],
            root_state_local=ground_root.tolist(),joint_position_rad=data.joint_pos[:,env.joint_ids].cpu().tolist(),
            joint_velocity_rad_s=data.joint_vel[:,env.joint_ids].cpu().tolist(),all_episode_clocks_zero=True))
        if episode_trace is not None:
            from isaacsim.core.utils.stage import get_current_stage
            from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
            export_robot_geometry(get_current_stage(),env.robot,args.output)
            adapter.trace_recorder=EpisodeTraceRecorderV4(args.output/'trajectory_traces',args.num_envs,
                round(protocol['episode_seconds']/protocol['policy_dt_s'])+1,env.robot.body_names,
                episode_trace,dict(native_source_confirmed=True,source_kind='Actual native changing-policy TRAIN',
                    training_indices=indices,initialization=initialization,
                    protocol_sha256=contract['protocol_sha256'],asset_sha256=protocol['asset']['sha256'],
                    body_names=env.robot.body_names,joint_names=[env.robot.joint_names[j]for j in env.joint_ids],
                    default_joint_position_rad=adapter.default.cpu().tolist(),joint_limits_rad=adapter.limits.cpu().tolist(),
                    source_job_output=str(args.output),formal_acceptance=False))
            adapter.trace_contact_cursor=len(env.contact_records)
        initial_update=runner.current_learning_iteration;history=[];interrupted=False
        for _ in range(args.updates):
            adapter.trace_iteration=runner.current_learning_iteration
            train_actor=runner.current_learning_iteration>=configuration['actor_warmup_updates']
            for parameter in runner.alg.policy.actor.parameters():parameter.requires_grad_(train_actor)
            runner.alg.policy.log_std.requires_grad_(train_actor)
            completed_update_safe=False
            try:runner.learn(1,init_at_random_ep_len=False)
            except ResidualTrainingInterrupted:
                interrupted=True;break
            completed_update_safe=True
            rows,counts=joined_episodes(env,cases,indices,protocol)
            entry=dict(update=runner.current_learning_iteration,actor_warmup=not train_actor,
                actual_control_env_steps=adapter.total_control_env_steps,**counts,
                losses=runner.alg.last_update_stats,formal_acceptance=False)
            if env.live_alignment_reward is not None:
                entry['actual_paid_alignment_rewards']=env.live_alignment_reward.statistics()
            if adapter.trace_recorder is not None:
                try:adapter.trace_recorder.flush()
                except Exception as error:adapter.trace_recorder.disable(error)
                entry['trajectory_recording']=adapter.trace_recorder.statistics()
            history.append(entry);atomic_json(args.output/'progress.json',entry)
            atomic_json(args.output/'episodes.partial.json',rows)
            print(json.dumps(entry),flush=True)
            # Plans are diagnostics, not policy state. Flush before clearing
            # long TRAIN logs; per-lane planning state remains intact.
            atomic_json(args.output/f'plans_update_{runner.current_learning_iteration:04d}.json',adapter.controller.logs)
            adapter.controller.logs=[]
            if stopping:interrupted=True;break
        rows,counts=joined_episodes(env,cases,indices,protocol)
        public_sample=adapter.last_observation['policy'][:16].detach().clone()
        runner.save_restart(args.output/'checkpoint.pt',infos=dict(residual_contract=contract,
            learner_initialization=initialization))
        if adapter.trace_recorder is not None:
            try:adapter.trace_recorder.flush()
            except Exception as error:adapter.trace_recorder.disable(error)
        policy=export_residual_policy(base,runner.alg.policy,scales,public_sample,
            dict(native_execution=adapter.total_control_env_steps>0,completed_updates=runner.current_learning_iteration,
                 base_policy_sha256=contract['base_policy_sha256'],actual_control_env_steps=adapter.total_control_env_steps,
                 **({'base_controller_mode':mode} if mode=='frozen_public_policy' else {}),
                 num_envs=args.num_envs,initialization=initialization))
        atomic_torch(args.output/'policy.pt',policy)
        atomic_json(args.output/'episodes.json',rows)
        atomic_json(args.output/'contacts.json',env.contact_records)
        atomic_json(args.output/'metrics.json',dict(scope='Changing-policy native TRAIN rollouts only; evaluate exported policy separately.',
            completed_updates=runner.current_learning_iteration-initial_update,history=history,counts=counts,initialization=initialization,
            checkpoint_aborted_episodes=env.checkpoint_aborted_episodes,
            policy_sha256=digest(args.output/'policy.pt'),formal_acceptance=False))
        if interrupted:
            atomic_json(args.output/'interrupted.json',dict(env_steps=adapter.total_control_env_steps,
                resume_from='checkpoint.pt',canonical_ground_restart=True,formal_acceptance=False))
            raise SystemExit(2)
        assert runner.current_learning_iteration-initial_update==args.updates
        atomic_json(args.output/'completed.json',dict(env_steps=adapter.total_control_env_steps))
    except BaseException as error:
        if runner is not None and completed_update_safe:
            try:
                save_completed_update_error_checkpoint(runner,args.output,contract,initialization,
                    error,adapter.total_control_env_steps,completed_update_safe)
            except BaseException as save_error:
                atomic_json(args.output/'error_recovery_checkpoint_failed.json',dict(
                    original_error=repr(error),save_error=repr(save_error),job_succeeded=False))
        raise
    finally:
        adapter.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:app.close(skip_cleanup=True)
