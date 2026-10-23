#!/usr/bin/env python3
"""Search high-arm poses for attack speed, outgoing direction and landing."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--seed',type=int,default=613080);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32)
Q0=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722]);QC=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977]);QD=torch.tensor([0.,1.,-1.,0.,0.,0.]);Q4=torch.tensor([-1.89106,1.20556,-2.65668,.86670,-.88788,-.27291]);Q1=torch.tensor([1.87380,1.33749,-2.84978,.20563,-.97821,-.11784])
try:
 a.output.mkdir(parents=True,exist_ok=True);g=torch.Generator(device='cuda').manual_seed(a.seed);cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=2.;cfg.shuttle_start=(8.18,0.,1.15);cfg.shuttle_velocity=(-6.6,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.);cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=P.to(env.device).repeat(a.num_envs,1);default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone();bases=torch.stack([QC,QD,Q4,Q1]).to(env.device);bi=torch.multinomial(torch.tensor([.55,.15,.15,.15],device=env.device),a.num_envs,replacement=True,generator=g);late=bases[bi]+.22*torch.randn((a.num_envs,6),generator=g,device=env.device);late[:,1]=late[:,1].clamp(0,3.14);late[:,2]=late[:,2].clamp(-2.967,0);late[:,4]=late[:,4].clamp(-1.22,1.22);late[:,5]=late[:,5].clamp(-2.094,2.094);switch=.98+.28*torch.rand(a.num_envs,generator=g,device=env.device);pre=(Q0.to(env.device)-default)/2.;late_action=(late-default[None,:])/2.;obs,_=w.reset();out_v=torch.zeros((a.num_envs,3),device=env.device);got=torch.zeros(a.num_envs,dtype=torch.bool,device=env.device)
 with torch.inference_mode():
  for _ in range(400):
   action=teacher_action(obs['policy'],params);t=obs['policy'][:,775];arm=torch.zeros((a.num_envs,6),device=env.device);active=(t>=.70)&(t<1.42);arm[active]=torch.where((t[active]<switch[active])[:,None],pre[None,:],late_action[active]);action[:,12:]=arm;obs,_,_,_=w.step(action);new=env.contact_seen&(~got);out_v[new]=env.shuttle.data.root_lin_vel_w[new];got|=new
 rows=[]
 for i,r in enumerate(env.ball_episode_records): rows.append({'env_id':i,'base_id':int(bi[i]),'switch_s':float(switch[i]),'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'outgoing_vx_mps':float(out_v[i,0]),'outgoing_vy_mps':float(out_v[i,1]),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz'],'late_joint_position':late[i].cpu().tolist()})
 score=lambda v:(v['contact'] and v['airborne_contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 and abs(v['outgoing_vy_mps'])<.6 and v['outgoing_vx_mps']>.2)
 m={'physics_dt_s':a.physics_dt,'episodes':len(rows),'contact':sum(v['contact'] for v in rows),'airborne_contact':sum(v['airborne_contact'] for v in rows),'attack':sum(v['contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 for v in rows),'directional_attack':sum(score(v) for v in rows),'crossed_net':sum(v['crossed_net'] for v in rows),'landed_target':sum(v['landed_target'] for v in rows),'recovery':sum(v['recovery_success'] for v in rows),'max_speed':max(v['outgoing_speed_mps'] for v in rows),'variants':rows};(a.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');(a.output/'variants.json').write_text(json.dumps({'base_names':['QC','QD','Q4','Q1'],'late_joint_position':late.cpu().tolist(),'switch_s':switch.cpu().tolist()},indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f: wr=csv.DictWriter(f,fieldnames=sorted(rows[0]));wr.writeheader();wr.writerows(rows)
 print(json.dumps({k:m[k] for k in ('episodes','contact','airborne_contact','attack','directional_attack','crossed_net','landed_target','recovery','max_speed')},sort_keys=True),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
