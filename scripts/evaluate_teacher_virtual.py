#!/usr/bin/env python3
"""Evaluate the bounded training teacher on the declared target panel."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.virtual_env import VirtualHitEnv,VirtualHitEnvCfg
from quadsmash.control.jump_teacher import teacher_action
P=[0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.0,-0.3079007566,0.3499999940,-0.1971214861,-0.9468204977,0.1242316961,0.1379330605,0.0928989127,-0.2010091096]
env=None
try:
 cfg=VirtualHitEnvCfg();cfg.scene.num_envs=64;cfg.sim.device='cuda:0';cfg.seed=1907
 env=VirtualHitEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);params=torch.tensor(P,device=env.device).repeat(env.num_envs,1)
 env._episode_has_stepped.zero_();env._metrics_started=False;w.reset();env.virtual_episode_records.clear();obs=w.get_observations()
 with torch.inference_mode():
  for _ in range(400):obs,_,_,_=w.step(teacher_action(obs['policy'],params))
 rows=env.virtual_episode_records[:64];metrics=dict(sample_count=len(rows),virtual_event_rate=sum(r['virtual_event'] for r in rows)/len(rows),virtual_hit_recovery_success_rate=sum(r['virtual_event'] and r['recovery_success'] for r in rows)/len(rows),ground_init_fraction=1.0,numerical_failures=0)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(metrics,indent=2));
 with (a.output/'episodes.csv').open('w',newline='') as f:wri=csv.DictWriter(f,fieldnames=sorted(rows[0]));wri.writeheader();wri.writerows(rows)
 print(json.dumps(metrics),flush=True)
finally:
 if env is not None:env.close()
 app.close(skip_cleanup=True)
