#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--shuttle-vz',type=float,default=7.0);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.0,-0.3079007566,0.3499999940,-0.1971214861,-0.9468204977,0.25,0.25,0.15,-0.20],dtype=torch.float32)
Q=torch.tensor([0.13985,1.47213,-2.90887,1.29097,-1.17635,-0.01722],dtype=torch.float32)
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=1;cfg.seed=613020;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True;cfg.episode_length_s=1.8;cfg.arm_action_scale=2.;cfg.shuttle_velocity=(-6.,0.,a.shuttle_vz);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(1,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]];arm=((Q.to(env.device)-default)/2.).view(1,6);obs,_=w.reset();rows=[]
 with torch.inference_mode():
  for tick in range(400):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];action[:,12:]=torch.where(((t>=.7)&(t<1.38))[:,None],arm,torch.zeros_like(arm))
   rid=env.racket_body_id;rows.append({'time_s':tick*.01,'ball_xyz':(env.shuttle.data.root_pos_w[0]-env.scene.env_origins[0]).cpu().tolist(),'ball_v':env.shuttle.data.root_lin_vel_w[0].cpu().tolist(),'tool_xyz':(env.robot.data.body_pos_w[0,rid]-env.scene.env_origins[0]).cpu().tolist(),'tool_v':env.robot.data.body_lin_vel_w[0,rid].cpu().tolist(),'tool_quat':env.robot.data.body_quat_w[0,rid].cpu().tolist(),'joint_pos':env.robot.data.joint_pos[0,env.joint_ids[12:]].cpu().tolist(),'contact':bool(env.contact_seen[0]),'airborne':bool(env.observed_airborne[0])});obs,_,_,_=w.step(action)
 (a.output/'trajectory.json').write_text(json.dumps(rows,indent=2));(a.output/'episode.json').write_text(json.dumps(env.ball_episode_records[-1] if env.ball_episode_records else {},indent=2));print(json.dumps(env.ball_episode_records[-1] if env.ball_episode_records else {},indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
