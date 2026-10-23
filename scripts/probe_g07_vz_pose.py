#!/usr/bin/env python3
"""Probe a bounded physical high-arm pose while varying shuttle launch height."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--vx',type=float,default=-6.6);p.add_argument('--vz',type=float,default=6.82);p.add_argument('--start-x',type=float,default=8.18);p.add_argument('--switch',type=float,default=1.18);p.add_argument('--alpha',type=float,default=0.0);p.add_argument('--pose',choices=('q0','q4','qc','q1'),default='qc');AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32)
Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);QC=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977]);Q4=torch.tensor([-1.89106,1.20556,-2.65668,.86670,-.88788,-.27291]);Q1=torch.tensor([1.87380,1.33749,-2.84978,.20563,-.97821,-.11784])
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=613090;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=2.;cfg.shuttle_start=(a.start_x,0.,1.15);cfg.shuttle_velocity=(a.vx,0.,a.vz);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.);cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(a.num_envs,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();target={'q0':Q0,'q4':Q4,'qc':QC,'q1':Q1}[a.pose].to(env.device);late=Q0.to(env.device) if a.pose=='q0' else ((1-a.alpha)*QC.to(env.device)+a.alpha*target);pre=(Q0.to(env.device)-default)/2.;late_action=(late-default)/2.;obs,_=w.reset()
 with torch.inference_mode():
  for _ in range(400):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];arm=torch.zeros((a.num_envs,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<a.switch)[:,None],pre[None,:],late_action[None,:]);action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=[]
 for i,r in enumerate(env.ball_episode_records): rows.append({'env_id':i,'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz']})
 m={'physics_dt_s':a.physics_dt,'vx':a.vx,'vz':a.vz,'start_x':a.start_x,'alpha':a.alpha,'pose':a.pose,'switch_s':a.switch,'episodes':len(rows),'contact':sum(v['contact'] for v in rows),'airborne_contact':sum(v['airborne_contact'] for v in rows),'attack':sum(v['contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 for v in rows),'crossed_net':sum(v['crossed_net'] for v in rows),'landed_target':sum(v['landed_target'] for v in rows),'recovery':sum(v['recovery_success'] for v in rows),'max_speed':max(v['outgoing_speed_mps'] for v in rows),'min_vz':min(v['outgoing_vz_mps'] for v in rows),'variants':rows};(a.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
 with (a.output/'episodes.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=sorted(rows[0]));wr.writeheader();wr.writerows(rows)
 print(json.dumps({k:m[k] for k in ('episodes','contact','airborne_contact','attack','crossed_net','landed_target','recovery','max_speed','min_vz')},sort_keys=True),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
