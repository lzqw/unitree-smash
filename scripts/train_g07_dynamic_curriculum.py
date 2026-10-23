#!/usr/bin/env python3
"""Learn complete near-net actions by BC/DAgger, then measure native rollouts.

The actor controls every policy step from reset, with no teacher prefix. These
are development curriculum results, not canonical G07 acceptance evidence.
"""
import argparse
import hashlib
import json
import signal
import sys
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=64)
parser.add_argument('--updates',type=int,default=2000)
parser.add_argument('--rounds',type=int,default=6)
parser.add_argument('--seed',type=int,default=714601)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app

import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv

stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)

def summarize(rows):
    air=sum(r['airborne_contact'] for r in rows)
    recovered=sum(r['airborne_contact'] and r['recovery_success'] for r in rows)
    return dict(episodes=len(rows),airborne_contact=air,
        ordered_net=sum(r['ordered_crossed_net'] for r in rows),
        ordered_landing=sum(r['ordered_landed_target'] for r in rows),
        recovery=sum(r['recovery_success'] for r in rows),
        conditional_recovery_rate=recovered/max(1,air),
        complete=sum(r['complete_smash'] for r in rows),
        numerical_failures=sum(r['numerical_failures'] for r in rows))

def main():
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    candidate=torch.load(args.candidate,map_location='cpu',weights_only=False)
    base,launch=candidate['base'],candidate['launch']
    cfg=BallEnvCfg()
    cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed
    cfg.sim.device='cuda:0'
    cfg.sim.dt,cfg.decimation,cfg.sim.render_interval=.00025,40,40
    cfg.episode_length_s=4.
    cfg.include_net=True
    cfg.robot.init_state.pos=(float(base['root_start_x_m']),0.,.323)
    cfg.shuttle_start=tuple(float(v) for v in launch['shuttle_start'])
    cfg.shuttle_velocity=tuple(float(v) for v in launch['shuttle_velocity'])
    cfg.event_goal_position=tuple(float(v) for v in base['virtual_ball_center_m'])
    cfg.event_goal_normal=tuple(float(v) for v in base['face_normal_world'])
    cfg.event_goal_time_s=float(launch['arrival_time_s'])
    env=FullJointSmashEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    runner_cfg=json.loads((ROOT/'configs/g07_high_arm.json').read_text())['runner']
    runner_cfg['seed']=args.seed
    runner_cfg['method']='absolute-joint BC/DAgger near-net curriculum from scratch'
    runner=NativeRunner(wrapped,runner_cfg,log_dir=str(args.output/'runner'),device=env.device)
    policy=runner.alg.policy
    params=candidate['parameters'].to(env.device).repeat(args.num_envs,1)
    arm=candidate['arm_pose'].to(env.device)
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    data=torch.load(args.dataset,map_location='cpu',weights_only=False)
    # Validation episodes are disjoint from training episodes, but use the same
    # development launch. This is not held-out task generalization.
    validation=torch.arange(data['observations'].shape[1])%5==0
    train_x=data['observations'][:,~validation].reshape(-1,794).to(env.device)
    train_y=data['actions'][:,~validation].reshape(-1,18).to(env.device)
    val_x=data['observations'][:,validation].reshape(-1,794).to(env.device)
    val_y=data['actions'][:,validation].reshape(-1,18).to(env.device)
    teacher_summary=summarize(data['episodes'])
    del data
    policy.train()
    policy.update_normalization(TensorDict({'policy':train_x},batch_size=[len(train_x)]))
    policy.eval()
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
    report=dict(kind='learned actor BC/DAgger development curriculum',
        formal_policy_acceptance=False,root_start_x_m=base['root_start_x_m'],
        teacher=teacher_summary,teacher_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        initial_weights='random; changed absolute-joint action contract',
        teacher_prefix_steps=0,learner_checks=[],env_steps=0)
    best=None
    try:
        for rnd in range(args.rounds):
            active=torch.where((train_x[:,775]>=.65)&(train_x[:,775]<=2.2))[0]
            losses=[]
            for update in range(args.updates):
                idx=torch.cat((active[torch.randint(len(active),(1024,),device=env.device)],
                               torch.randint(len(train_x),(1024,),device=env.device)))
                obs=TensorDict({'policy':train_x[idx]},batch_size=[len(idx)])
                prediction=policy.act_inference(obs)
                loss=(prediction-train_y[idx]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
                optimizer.step()
                if update%250==0:
                    losses.append(float(loss))
                    print(json.dumps(dict(round=rnd,update=update,loss=losses[-1])),flush=True)
                if stopping:break
            with torch.no_grad():
                validation_mse=float((policy.act_inference(TensorDict({'policy':val_x},batch_size=[len(val_x)]))-val_y).square().mean())
            env._episode_has_stepped.zero_()
            env._metrics_started=False
            env.episode_records.clear()
            env.ball_episode_records.clear()
            env.smash_episode_records.clear()
            obs,_=wrapped.reset()
            dx,dy=[],[]
            with torch.inference_mode():
                for _ in range(400):
                    dx.append(obs['policy'].clone())
                    dy.append(dynamic_reach_action(obs,params,default,limits,arm))
                    action=policy.act_inference(obs)
                    if not torch.isfinite(action).all():raise RuntimeError('Nonfinite learned action')
                    obs,_,_,_=wrapped.step(action)
            first={}
            for row in env.smash_episode_records:first.setdefault(row['env_id'],row)
            if len(first)!=args.num_envs:raise RuntimeError('Incomplete actor evaluation')
            rows=[first[i] for i in range(args.num_envs)]
            check=dict(round=rnd,validation_mse=validation_mse,training_losses=losses,**summarize(rows))
            report['learner_checks'].append(check)
            report['env_steps']+=400*args.num_envs
            (args.output/f'episodes_{rnd:02d}.json').write_text(json.dumps(rows,indent=2)+'\n')
            # A saved actor always restarts at ground reset, never a teacher state.
            runner.save_restart(args.output/f'bc_{rnd:02d}.pt',infos=dict(curriculum=True,round=rnd,action_contract=env.action_family))
            key=(check['complete'],check['ordered_landing'],check['ordered_net'],check['airborne_contact'],check['recovery'],-check['numerical_failures'])
            if best is None or key>best:
                best=key
                runner.save_restart(args.output/'checkpoint.pt',infos=dict(curriculum=True,round=rnd,action_contract=env.action_family))
                report['best_round']=rnd
            torch.save(dict(optimizer=optimizer.state_dict(),round=rnd,seed=args.seed),args.output/'bc_optimizer.pt')
            (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
            print('LEARNER='+json.dumps(check),flush=True)
            xx,yy=torch.stack(dx).clone(),torch.stack(dy).clone()
            keep=torch.arange(args.num_envs,device=env.device)%5!=0
            train_x=torch.cat((train_x,xx[:,keep].reshape(-1,794)))
            train_y=torch.cat((train_y,yy[:,keep].reshape(-1,18)))
            torch.save(dict(observations=xx.cpu(),actions=yy.cpu()),args.output/f'dagger_{rnd:02d}.pt')
            del dx,dy,xx,yy
            if stopping:break
        if len(report['learner_checks'])==args.rounds and not stopping:
            (args.output/'completed.json').write_text(json.dumps(dict(rounds=args.rounds,env_steps=report['env_steps']))+'\n')
    finally:
        env.close()

try:
    main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True)
    error=traceback.format_exc()
    (args.output/'error.txt').write_text(error)
    print(error,file=sys.stderr,flush=True)
    raise
finally:
    app.close(skip_cleanup=True)
