#!/usr/bin/env python3
"""Search one physical shuttle launch around the best high-arm trajectory."""
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--num-envs',type=int,default=64); p.add_argument('--physics-dt',type=float,default=.00025); p.add_argument('--seed',type=int,default=613043); p.add_argument('--shuttle-vx',type=float,default=-6.2); p.add_argument('--shuttle-vz',type=float,default=6.82); p.add_argument('--shuttle-start-x',type=float,default=7.4); p.add_argument('--switch',type=float,default=1.18); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32)
Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]); QC=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977])
try:
 a.output.mkdir(parents=True,exist_ok=True); cfg=BallEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.device='cuda:0'; cfg.sim.dt=a.physics_dt; cfg.decimation=round(.01/a.physics_dt); cfg.sim.render_interval=cfg.decimation; cfg.include_net=True; cfg.episode_length_s=1.8; cfg.arm_action_scale=2.; cfg.shuttle_start=(a.shuttle_start_x,0.,1.15); cfg.shuttle_velocity=(a.shuttle_vx,0.,a.shuttle_vz); cfg.event_goal_position=(.3,0.,1.25); cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg); w=RslRlVecEnvWrapper(env,clip_actions=None); params=P.to(env.device).repeat(env.num_envs,1); default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone(); pre=(Q0.to(env.device)-default)/2.; late=(QC.to(env.device)-default)/2.; obs,_=w.reset()
 with torch.inference_mode():
  for _ in range(182):
   action=teacher_action(obs['policy'],params); t=obs['policy'][:,775]; arm=torch.zeros((a.num_envs,6),device=env.device); active=(t>=.70)&(t<1.42); arm[active]=torch.where((t[active]<a.switch)[:,None],pre[None,:],late[None,:]); action[:,12:]=arm; obs,_,_,_=w.step(action)
 rows=[]
 for i,r in enumerate(list(env.ball_episode_records)): rows.append({'env_id':i,'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz']})
 m={'physics_dt_s':a.physics_dt,'shuttle_start_x':a.shuttle_start_x,'shuttle_velocity':[a.shuttle_vx,0.,a.shuttle_vz],'switch_s':a.switch,'arm_action_scale':2.,'episodes':len(rows),'contact':sum(v['contact'] for v in rows),'attack':sum(v['contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 for v in rows),'max_speed':max(v['outgoing_speed_mps'] for v in rows),'min_vz':min(v['outgoing_vz_mps'] for v in rows),'variants':rows}; (a.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=sorted(rows[0])); wr.writeheader(); wr.writerows(rows)
 print(json.dumps({k:m[k] for k in ('contact','attack','max_speed','min_vz')},sort_keys=True),flush=True)
finally:
 try: env.close()
 except: pass
 app.close(skip_cleanup=True)
