#!/usr/bin/env python3
"""Search high-arm prepose-to-swing schedules in real PhysX."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=8);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--shuttle-vz',type=float,default=6.82);p.add_argument('--shuttle-vx',type=float,default=-6.0);p.add_argument('--shuttle-start-x',type=float,default=7.4);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.,-0.3079007566,0.349999994,-0.1971214861,-0.9468204977,.25,.25,.15,-.20],dtype=torch.float32)
Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);Q1=torch.tensor([1.87380,1.33749,-2.84978,.20563,-.97821,-.11784]);Q2=torch.tensor([.35977,1.27435,-2.55655,-1.72646,-1.17798,-.19464]);Q4=torch.tensor([-1.89106,1.20556,-2.65668,.86670,-.88788,-.27291]);QD=torch.tensor([0.,1.,-1.,0.,0.,0.]);QL=torch.tensor([0.,.8,-1.5,0.,0.,0.])
# (prepose, swing pose, switch time); all targets are bounded by arm_action_scale=2.
V=[('to_default_1p15',Q0,QD,1.15),('to_default_1p20',Q0,QD,1.20),('to_default_1p25',Q0,QD,1.25),('to_q1_1p20',Q0,Q1,1.20),('to_q1_1p25',Q0,Q1,1.25),('to_q4_1p15',Q0,Q4,1.15),('to_q4_1p20',Q0,Q4,1.20),('to_q2_1p20',Q0,Q2,1.20)]
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=613030;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True;cfg.episode_length_s=1.8;cfg.arm_action_scale=2.;cfg.shuttle_start=(a.shuttle_start_x,0.,1.15);cfg.shuttle_velocity=(a.shuttle_vx,0.,a.shuttle_vz);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(env.num_envs,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();pre=torch.stack([v[1] for v in V]).to(env.device);late=torch.stack([v[2] for v in V]).to(env.device);pre=(pre-default)/2.;late=(late-default)/2.;switch=torch.tensor([v[3] for v in V],device=env.device);obs,_=w.reset();ids=torch.arange(a.num_envs,device=env.device);vid=torch.clamp(ids,max=len(V)-1)
 with torch.inference_mode():
  for _ in range(182):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];arm=torch.zeros((a.num_envs,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<switch[vid[active]])[:,None],pre[vid[active]],late[vid[active]]);action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=list(env.ball_episode_records);out=[]
 for i,v in enumerate(V):
  r=rows[i];out.append({'variant':v[0],'switch_s':v[3],'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz']})
 (a.output/'metrics.json').write_text(json.dumps({'physics_dt_s':a.physics_dt,'shuttle_velocity':[a.shuttle_vx,0,a.shuttle_vz],'shuttle_start_x':a.shuttle_start_x,'arm_action_scale':2.,'variants':out},indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=sorted(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(out,indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
