#!/usr/bin/env python3
"""Search bounded physical arm action schedules for a real-net smash probe."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--switch',type=float,default=1.10);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
names=['zero','pos_to_neg','neg_to_pos','diag_to_opp','opp_to_diag','wrist_to_opp','legswing_to_opp','mixed_to_zero']
# Early arm target followed by a brief pre-contact swing target; after 1.45 s the learner arm is retained.
variants=[([0]*6,[0]*6),([1]*6,[-1]*6),([-1]*6,[1]*6),([1,-1,1,-1,1,-1],[-1,1,-1,1,-1,1]),([-1,1,-1,1,-1,1],[1,-1,1,-1,1,-1]),([0,1,-1,0,1,-1],[0,-1,1,0,-1,1]),([1,1,-1,-1,1,-1],[-1,-1,1,1,-1,1]),([-1,-1,1,1,-1,1],[0,0,0,0,0,0])]
try:
 a.output.mkdir(parents=True,exist_ok=True)
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=611050;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=1.0);config=json.loads((ROOT/'configs/g05_imitation.json').read_text());runner=OnPolicyRunner(w,config['runner'],log_dir=str(a.output/'ppo'),device=env.device);runner.load(str(a.checkpoint.resolve()),load_optimizer=False);policy=runner.get_inference_policy(device=env.device)
 obs,_=w.reset();n=a.num_envs//len(variants); ids=torch.arange(a.num_envs,device=env.device);vix=torch.clamp(ids//n,max=len(variants)-1)
 with torch.inference_mode():
  for _ in range(400):
   action=policy(obs); t=obs['policy'][:,775]; arm=torch.zeros((a.num_envs,6),device=env.device)
   for i,(early,late) in enumerate(variants):
    mask=vix==i; e=torch.tensor(early,device=env.device,dtype=torch.float32); l=torch.tensor(late,device=env.device,dtype=torch.float32)
    active=(t[mask]>=a.switch)&(t[mask]<1.45)
    arm[mask]=torch.where(active[:,None],l,e)
    post=t[mask]>=1.45
    arm[mask]=torch.where(post[:,None],action[mask,12:],arm[mask])
   action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=list(env.ball_episode_records);out=[]
 for i,name in enumerate(names):
  sub=rows[i*n:(i+1)*n];N=max(len(sub),1);contact=sum(bool(r['contact']) for r in sub);air=sum(bool(r['airborne_contact']) for r in sub);attack=sum(bool(r['contact'] and float(r['outgoing_speed_mps'])>=8 and float(r['outgoing_vz_mps'])<=-1) for r in sub);complete=sum(bool(r['contact'] and float(r['outgoing_speed_mps'])>=8 and float(r['outgoing_vz_mps'])<=-1 and r['crossed_net'] and r['landed_target'] and r['recovery_success']) for r in sub);out.append({'variant':name,'episodes':len(sub),'contact':contact,'airborne_contact':air,'attack':attack,'crossed_net':sum(bool(r['crossed_net']) for r in sub),'landed_target':sum(bool(r['landed_target']) for r in sub),'recovery':sum(bool(r['recovery_success']) for r in sub),'complete':complete,'mean_outgoing_speed':sum(float(r['outgoing_speed_mps']) for r in sub)/N,'mean_outgoing_vz':sum(float(r['outgoing_vz_mps']) for r in sub)/N,'body_contact':sum(bool(r['body_contact']) for r in sub)})
 (a.output/'metrics.json').write_text(json.dumps({'physics_dt_s':a.physics_dt,'variants':out},indent=2)+'\n');
 with (a.output/'episodes.csv').open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=sorted(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(out,indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
