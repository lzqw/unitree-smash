#!/usr/bin/env python3
"""Replay two fixed real-PhysX candidate poses and record full shuttle flight."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-dt',type=float,default=.00025);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32);Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);
POSES=torch.tensor([[-.198686,1.437229,-1.356817,.348973,1.220000,-.573214],[.127227,2.498345,-1.046097,.287897,.795409,-.377202]],dtype=torch.float32);SW=torch.tensor([1.212954,1.287992])
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=2;cfg.seed=613097;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=2.;cfg.shuttle_start=(8.18,0.,1.15);cfg.shuttle_velocity=(-6.6,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.);cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(2,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();pre=(Q0.to(env.device)-default)/2.;late=(POSES.to(env.device)-default[None,:])/2.;sw=SW.to(env.device);obs,_=w.reset();rows=[[] for _ in range(2)]
 with torch.inference_mode():
  for tick in range(500):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];arm=torch.zeros((2,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<sw[active])[:,None],pre[None,:],late[active]);action[:,12:]=arm
   b=env.shuttle.data.root_pos_w-env.scene.env_origins;v=env.shuttle.data.root_lin_vel_w;root=env.robot.data.root_pos_w-env.scene.env_origins
   for i in range(2): rows[i].append({'time_s':tick*.01,'ball_xyz':b[i].cpu().tolist(),'ball_v':v[i].cpu().tolist(),'root_xyz':root[i].cpu().tolist(),'contact':bool(env.contact_seen[i]),'crossed_net':bool(env.crossed_net[i]),'landed_target':bool(env.landed_target[i]),'landed':bool(env.landed[i])})
   obs,_,_,_=w.step(action)
 (a.output/'trajectory.json').write_text(json.dumps(rows,indent=2));(a.output/'episodes.json').write_text(json.dumps(env.ball_episode_records,indent=2));print(json.dumps(env.ball_episode_records,indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
