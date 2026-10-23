#!/usr/bin/env python3
"""Vectorized continuous high-arm swing search with real PhysX contact."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--seed',type=int,default=613040);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.,-0.3079007566,0.349999994,-0.1971214861,-.9468204977,.25,.25,.15,-.20],dtype=torch.float32)
Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);QD=torch.tensor([0.,1.,-1.,0.,0.,0.]);Q4=torch.tensor([-1.89106,1.20556,-2.65668,.86670,-.88788,-.27291]);Q1=torch.tensor([1.87380,1.33749,-2.84978,.20563,-.97821,-.11784])
try:
 a.output.mkdir(parents=True,exist_ok=True);g=torch.Generator(device='cuda').manual_seed(a.seed)
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True;cfg.episode_length_s=1.8;cfg.arm_action_scale=2.;cfg.shuttle_velocity=(-6.,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(env.num_envs,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();
 # Interpolate each candidate from the high prepose toward several lower/side poses, with bounded per-joint perturbation.
 bases=torch.stack([QD,Q4,Q1]).to(env.device);bi=torch.randint(0,3,(a.num_envs,),generator=g,device=env.device);base=bases[bi];alpha=torch.rand((a.num_envs,1),generator=g,device=env.device)*.95+.05;late=default[None,:]+alpha*(base-default[None,:]);late += .12*torch.randn((a.num_envs,6),generator=g,device=env.device);late[:,1]=late[:,1].clamp(0,3.14);late[:,2]=late[:,2].clamp(-2.967,0);late[:,4]=late[:,4].clamp(-1.22,1.22);late[:,5]=late[:,5].clamp(-2.094,2.094);late_action=(late-default[None,:])/2.;pre_action=(Q0.to(env.device)-default[None,:])/2.;switch=.98+.30*torch.rand(a.num_envs,generator=g,device=env.device)
 (a.output/'variants.json').write_text(json.dumps({'seed':a.seed,'late_joint_position':late.cpu().tolist(),'switch_s':switch.cpu().tolist()},indent=2)+'\n')
 obs,_=w.reset();ids=torch.arange(a.num_envs,device=env.device)
 with torch.inference_mode():
  for _ in range(182):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];arm=torch.zeros((a.num_envs,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<switch[active])[:,None],pre_action[None,:],late_action[active]);action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=list(env.ball_episode_records);out=[]
 for i,r in enumerate(rows): out.append({'env_id':i,'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz']})
 (a.output/'metrics.json').write_text(json.dumps({'physics_dt_s':a.physics_dt,'arm_action_scale':2.,'episodes':len(out),'contact':sum(v['contact'] for v in out),'attack':sum(v['contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 for v in out),'max_speed':max(v['outgoing_speed_mps'] for v in out),'min_vz':min(v['outgoing_vz_mps'] for v in out),'variants':out},indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=sorted(out[0]));writer.writeheader();writer.writerows(out)
 print(json.dumps(json.loads((a.output/'metrics.json').read_text()),indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
