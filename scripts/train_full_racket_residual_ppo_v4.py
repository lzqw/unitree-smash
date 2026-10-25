#!/usr/bin/env python3
"""Learn bounded whole-body residuals on the original full-model TRAIN suite."""
import argparse,copy,hashlib,json,signal,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--base-checkpoint',type=Path,required=True)
p.add_argument('--config',type=Path,default=ROOT/'configs/g07_full_racket_residual_ppo_v4.json')
p.add_argument('--resume',type=Path)
p.add_argument('--resize-ground',action='store_true',help='Explicitly fork a canonical learner onto a larger fresh ground batch; no old physical state restoration')
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--updates',type=int,default=8)
p.add_argument('--seed',type=int,default=748501)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.full_racket_residual_env_v4 import FullRacketResidualTrainingEnvV4
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_residual_reward_v4 import EVENT_WEIGHTS
from quadsmash.control.measured_joint_placement_reward_v4 import placement_settings,REWARD_VERSION
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.rl.full_racket_residual_ppo_v4 import (
    ResidualRslAdapterV4,ResidualTrainingInterrupted,export_residual_policy)
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
    if args.resize_ground and args.resume is None:raise ValueError('--resize-ground requires an actual canonical --resume checkpoint')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol,None)
    if args.num_envs<len(cases) or args.num_envs%len(cases):
        raise ValueError('Use a positive multiple of all128 fixed TRAIN cases')
    base=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False)
    if not base.get('native_execution') or base['schema']!='full-racket-ball-cem-v4':
        raise ValueError('Require one actually measured baseline controller, no inherited residual actor')
    if base['seed'].get('arm_target_rate_limit_rad_s')!=5.:
        raise ValueError('Use a measured baseline with the same5rad/s arm target limiter')
    if base['asset_sha256']!=protocol['asset']['sha256'] or digest(ROOT/protocol['asset']['path'])!=protocol['asset']['sha256']:
        raise ValueError('Baseline/full-model asset mismatch')
    for name,expected in protocol['physical_configs'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Physical config changed: '+name)
    configuration=json.loads(args.config.read_text());scales=configuration['joint_scale_rad']
    joint_placement=placement_settings(configuration.get('joint_placement_reward'))
    contract=dict(base_policy_sha256=digest(args.base_checkpoint),protocol_sha256=digest(protocol_path),
        joint_scale_rad=scales,reward_version='native-full-chain-residual-v1',event_weights=EVENT_WEIGHTS,
        actor_warmup_updates=configuration['actor_warmup_updates'],seed=args.seed)
    if joint_placement is not None:
        contract.update(reward_version=REWARD_VERSION,joint_placement_reward=joint_placement)
    train_cfg=copy.deepcopy(configuration['runner']);train_cfg['seed']=args.seed;train_cfg['residual_contract']=contract
    if train_cfg['obs_groups']['policy']!=['policy'] or train_cfg['policy']['class_name']!='ActorCritic' or train_cfg['policy']['noise_std_type']!='log':
        raise ValueError('This adapter requires the declared public-only feedforward log-std actor')
    cfg=full_racket_cfg(args.num_envs,protocol['physics_dt_s']);cfg.seed=args.seed
    cfg.episode_length_s=protocol['episode_seconds']
    env=FullRacketResidualTrainingEnvV4(cfg,cases=cases*(args.num_envs//len(cases)),
        continue_after_miss=True,joint_placement_reward=joint_placement)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    adapter=ResidualRslAdapterV4(wrapped,base,scales,lambda:stopping)
    try:
        if env.robot.is_fixed_base:raise ValueError('Actual floating ground-start robot required')
        atomic_json(args.output/'asset_dependencies.json',audit_composed_asset(ROOT,cfg.robot.spawn.usd_path))
        atomic_json(args.output/'INPUTS.json',dict(contract=contract,training_indices=indices,
            num_envs=args.num_envs,requested_updates=args.updates,runner=train_cfg,
            configuration_sha256=digest(args.config),resume_checkpoint_sha256=digest(args.resume) if args.resume else None,
            resize_ground_requested=args.resize_ground,
            physics_dt_s=cfg.sim.dt,policy_dt_s=env.step_dt,formal_cases_read=False,
            formal_acceptance=False,scope='Native TRAIN learning across changing weights; no frozen-policy performance.'))
        runner=NativeRunner(adapter,train_cfg,log_dir=str(args.output/'ppo'),device=env.device)
        initialization=dict(mode='fresh_zero_mean_actor',num_envs=args.num_envs,initial_iteration=0)
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
        else:
            output_layer=[m for m in runner.alg.policy.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
            with torch.no_grad():output_layer.weight.zero_();output_layer.bias.zero_()
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
        initial_update=runner.current_learning_iteration;history=[];interrupted=False
        for _ in range(args.updates):
            train_actor=runner.current_learning_iteration>=configuration['actor_warmup_updates']
            for parameter in runner.alg.policy.actor.parameters():parameter.requires_grad_(train_actor)
            runner.alg.policy.log_std.requires_grad_(train_actor)
            try:runner.learn(1,init_at_random_ep_len=False)
            except ResidualTrainingInterrupted:
                interrupted=True;break
            rows,counts=joined_episodes(env,cases,indices,protocol)
            entry=dict(update=runner.current_learning_iteration,actor_warmup=not train_actor,
                actual_control_env_steps=adapter.total_control_env_steps,**counts,
                losses=runner.alg.last_update_stats,formal_acceptance=False)
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
        runner.save_restart(args.output/'checkpoint.pt',infos=dict(residual_contract=contract))
        policy=export_residual_policy(base,runner.alg.policy,scales,public_sample,
            dict(native_execution=adapter.total_control_env_steps>0,completed_updates=runner.current_learning_iteration,
                 base_policy_sha256=contract['base_policy_sha256'],actual_control_env_steps=adapter.total_control_env_steps,
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
    finally:
        adapter.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:app.close(skip_cleanup=True)
