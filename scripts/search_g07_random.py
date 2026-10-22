#!/usr/bin/env python3
"""Vectorized search of bounded pre-contact arm target reversals."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--seed',type=int,default=611060);p.add_argument('--switch',type=float,default=None);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
try:
 a.output.mkdir(parents=True,exist_ok=True); g=torch.Generator(device='cuda').manual_seed(a.seed)
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.include_net=True;cfg.episode_length_s=1.8
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=1.0);config=json.loads((ROOT/'configs/g05_imitation.json').read_text());runner=OnPolicyRunner(w,config['runner'],log_dir=str(a.output/'ppo'),device=env.device);runner.load(str(a.checkpoint.resolve()),load_optimizer=False);policy=runner.get_inference_policy(device=env.device)
 # discrete bounded actions with deliberately broad timing; each env is one independent candidate.
 early=torch.randint(-1,2,(a.num_envs,6),generator=g,device=env.device).float();late=torch.randint(-1,2,(a.num_envs,6),generator=g,device=env.device).float();switch=torch.full((a.num_envs,),a.switch,device=env.device) if a.switch is not None else (0.85+0.45*torch.rand(a.num_envs,generator=g,device=env.device))
 (a.output/'variants.json').write_text(json.dumps({'seed':a.seed,'early':early.cpu().tolist(),'late':late.cpu().tolist(),'switch':switch.cpu().tolist()},indent=2)+'\n')
 obs,_=w.reset()
 with torch.inference_mode():
  for _ in range(round(cfg.episode_length_s/.01)+2):
   action=policy(obs);t=obs['policy'][:,775];arm=action[:,12:].clone();active=(t>=switch)&(t<1.45);arm=torch.where(active[:,None],late,torch.where((t<switch)[:,None],early,arm));action[:,12:]=arm;obs,_,_,_=w.step(action)
 rows=list(env.ball_episode_records);N=max(len(rows),1)
 with (a.output/'episodes.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=sorted(rows[0]));writer.writeheader();writer.writerows(rows)
 summary={'physics_dt_s':a.physics_dt,'episode_seconds':cfg.episode_length_s,'episodes':len(rows),'contact':sum(bool(r['contact']) for r in rows),'airborne_contact':sum(bool(r['airborne_contact']) for r in rows),'attack':sum(bool(r['contact'] and float(r['outgoing_speed_mps'])>=8 and float(r['outgoing_vz_mps'])<=-1) for r in rows),'max_speed':max(float(r['outgoing_speed_mps']) for r in rows),'min_vz':min(float(r['outgoing_vz_mps']) for r in rows),'mean_speed':sum(float(r['outgoing_speed_mps']) for r in rows)/N,'mean_vz':sum(float(r['outgoing_vz_mps']) for r in rows)/N,'crossed_net':sum(bool(r['crossed_net']) for r in rows),'landed_target':sum(bool(r['landed_target']) for r in rows),'recovery':sum(bool(r['recovery_success']) for r in rows),'body_contact':sum(bool(r['body_contact']) for r in rows)}
 (a.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
