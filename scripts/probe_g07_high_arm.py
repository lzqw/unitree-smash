#!/usr/bin/env python3
"""Probe high-arm, high-flight attack schedules in the real PhysX task."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=8);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--shuttle-vz',type=float,default=10.0);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.0,-0.3079007566,0.3499999940,-0.1971214861,-0.9468204977,0.25,0.25,0.15,-0.20],dtype=torch.float32)
# Full-range configurations from native joint-limit sampling, selected for high z and upward racket normal.
Q=[
 [0.13985,1.47213,-2.90887,1.29097,-1.17635,-0.01722],
 [1.87380,1.33749,-2.84978,0.20563,-0.97821,-0.11784],
 [0.35977,1.27435,-2.55655,-1.72646,-1.17798,-0.19464],
 [-1.08285,1.24934,-2.35607,1.38059,-0.99852,0.31309],
 [-1.89106,1.20556,-2.65668,0.86670,-0.88788,-0.27291],
 [-2.06765,1.20385,-2.66735,0.80115,-0.76081,0.38037],
 [1.82063,1.10119,-2.55646,1.72426,-0.92764,-0.28630],
 [-1.30231,1.40770,-2.57381,1.48893,-0.75636,0.91515],
]
names=[f'q{i}' for i in range(len(Q))]
try:
 a.output.mkdir(parents=True,exist_ok=True)
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=613010;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True;cfg.arm_action_scale=2.0;cfg.shuttle_velocity=(-6.0,0.0,a.shuttle_vz);cfg.event_goal_position=(0.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(env.num_envs,1);targets=torch.tensor(Q,device=env.device,dtype=torch.float32);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();actions=(targets-default[None,:])/cfg.arm_action_scale
 obs,_=w.reset(); ids=torch.arange(a.num_envs,device=env.device);vid=torch.clamp(ids,max=len(Q)-1)
 with torch.inference_mode():
  for _ in range(400):
   action=teacher_action(obs['policy'],params)
   t=obs['policy'][:,775]; arm=torch.zeros((a.num_envs,6),device=env.device)
   pre=(t>=.70)&(t<1.38); arm[pre]=actions[vid[pre]]
   action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=list(env.ball_episode_records);out=[]
 for i,name in enumerate(names):
  sub=rows[i:i+1];
  if not sub: continue
  r=sub[0];out.append({'variant':name,'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'contact_ball_xyz':r['ball_min_distance_ball_xyz'],'contact_tool_xyz':r['ball_min_distance_tool_xyz']})
 (a.output/'metrics.json').write_text(json.dumps({'shuttle_velocity':[-6,0,a.shuttle_vz],'arm_action_scale':cfg.arm_action_scale,'variants':out},indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=sorted(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(out,indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
