#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-dt',type=float,default=.00025);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
try:
 cfg=BallEnvCfg();cfg.scene.num_envs=1;cfg.seed=611040;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation
 env=BallEnv(cfg); w=RslRlVecEnvWrapper(env,clip_actions=1.0)
 config=json.loads((ROOT/'configs/g05_imitation.json').read_text()); runner=OnPolicyRunner(w,config['runner'],log_dir=str(a.output/'ppo'),device=env.device); runner.load(str(a.checkpoint.resolve()),load_optimizer=False); policy=runner.get_inference_policy(device=env.device)
 obs,_=w.reset(); rows=[]
 with torch.inference_mode():
  for tick in range(400):
   action=policy(obs)
   if tick%1==0:
    rid=env.racket_body_id; r=env.robot.data.body_pos_w[0,rid]-env.scene.env_origins[0]; rv=env.robot.data.body_lin_vel_w[0,rid]; b=env.shuttle.data.root_pos_w[0]-env.scene.env_origins[0];bv=env.shuttle.data.root_lin_vel_w[0]; q=env.robot.data.joint_pos[0]; jv=env.robot.data.joint_vel[0]
    rows.append({'tick':tick,'time_s':tick*.01,'action':action[0].cpu().tolist(),'racket_xyz':r.cpu().tolist(),'racket_v':rv.cpu().tolist(),'ball_xyz':b.cpu().tolist(),'ball_v':bv.cpu().tolist(),'joint_pos':q.cpu().tolist(),'joint_vel':jv.cpu().tolist(),'contact':bool(env.contact_seen[0]),'airborne':bool(env.observed_airborne[0])})
   obs,_,_,_=w.step(action)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'trajectory.json').write_text(json.dumps(rows,indent=2)); print(json.dumps({'rows':len(rows),'records':len(env.ball_episode_records),'contact':env.ball_episode_records[-1] if env.ball_episode_records else None},indent=2),flush=True)
finally:
 try: env.close()
 except: pass
 app.close(skip_cleanup=True)
