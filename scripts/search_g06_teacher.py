#!/usr/bin/env python3
"""Search bounded post-contact recovery parameters in real BallEnv."""
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--physics-dt',type=float,default=.00025);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action

BASE=[0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.0,
      -0.3079007566,0.3499999940,-0.1971214861,-0.9468204977,0.1242316961,
      0.1379330605,0.0928989127,-0.2010091096]
# (name, damping, recovery posture, front bias, pitch feedback)
VARIANTS=[('base',.124,.138,.093,-.201),('damp025',.25,.138,.093,-.201),
 ('damp035',.35,.138,.093,-.201),('post025',.20,.25,.093,-.201),
 ('post030',.12,.30,.093,-.201),('balanced',.25,.25,.15,-.2),
 ('front020',.25,.20,.20,-.2),('pitch_positive',.20,.20,.10,.35)]
env=None
try:
 a.output.mkdir(parents=True,exist_ok=True)
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=611030;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None)
 params=torch.tensor([BASE for _ in range(a.num_envs)],device=env.device)
 for idx,(name,damp,post,fbias,pgain) in enumerate(VARIANTS):
  lo,hi=idx*(a.num_envs//len(VARIANTS)),(idx+1)*(a.num_envs//len(VARIANTS))
  params[lo:hi,10]=damp;params[lo:hi,11]=post;params[lo:hi,12]=fbias;params[lo:hi,13]=pgain
 obs,_=w.reset();
 with torch.inference_mode():
  for _ in range(400): obs,_,_,_=w.step(teacher_action(obs['policy'],params))
 rows=list(env.ball_episode_records)
 out=[]
 for idx,(name,*_) in enumerate(VARIANTS):
  subset=rows[idx*(a.num_envs//len(VARIANTS)):(idx+1)*(a.num_envs//len(VARIANTS))]
  n=max(1,len(subset));out.append({'variant':name,'episodes':len(subset),'contacts':sum(bool(r['contact']) for r in subset),'airborne_contacts':sum(bool(r['airborne_contact']) for r in subset),'contact_recovery':sum(bool(r['contact'] and r['recovery_success']) for r in subset),'contact_recovery_rate':sum(bool(r['contact'] and r['recovery_success']) for r in subset)/n,'body_contacts':sum(bool(r['body_contact']) for r in subset),'mean_min_distance_m':sum(float(r['ball_min_distance_m']) for r in subset)/n})
 (a.output/'metrics.json').write_text(json.dumps({'physics_dt_s':a.physics_dt,'variants':out,'base_parameters':BASE},indent=2)+'\n')
 with (a.output/'episodes.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=sorted(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(out,indent=2),flush=True)
finally:
 if env is not None:env.close()
 app.close(skip_cleanup=True)
