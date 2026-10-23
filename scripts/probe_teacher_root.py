#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--leg-push',type=float,default=1.0);p.add_argument('--leg-crouch',type=float,default=.5);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32);Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);QC=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977])
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=1;cfg.seed=613091;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=2.;cfg.leg_action_scale_crouch=a.leg_crouch;cfg.leg_action_scale_push=a.leg_push;cfg.shuttle_start=(8.18,0.,1.15);cfg.shuttle_velocity=(-6.6,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();pre=(Q0.to(env.device)-default)/2.;late=(QC.to(env.device)-default)/2.;obs,_=w.reset();rows=[]
 with torch.inference_mode():
  for tick in range(400):
   action=teacher_action(obs['policy'],params[None]);t=obs['policy'][:,775];arm=torch.zeros((1,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<1.18)[:,None],pre[None,:],late[None,:]);action[:,12:]=arm;root=env.robot.data.root_pos_w[0].clone();tool=env.robot.data.body_pos_w[0,env.racket_body_id].clone();rows.append({'time_s':tick*.01,'root_xyz':(root-env.scene.env_origins[0]).cpu().tolist(),'tool_xyz':(tool-env.scene.env_origins[0]).cpu().tolist(),'airborne':bool(env.observed_airborne[0]),'action_legs':action[0,:12].cpu().tolist()});obs,_,_,_=w.step(action)
 (a.output/'trajectory.json').write_text(json.dumps(rows,indent=2));print('max_root_z',max(r['root_xyz'][2] for r in rows),'max_tool_z',max(r['tool_xyz'][2] for r in rows),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
