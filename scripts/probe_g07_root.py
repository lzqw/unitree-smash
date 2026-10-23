#!/usr/bin/env python3
"""Record robot root/racket and shuttle geometry for one frozen policy."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--physics-dt',type=float,default=.00025);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=1;cfg.seed=1908;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=2.;cfg.shuttle_start=(8.18,0.,1.15);cfg.shuttle_velocity=(-6.6,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.);cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=1.0);config=json.loads((ROOT/'configs/g07_high_attack.json').read_text());runner=OnPolicyRunner(w,config['runner'],log_dir=str(a.output/'ppo'),device=env.device);runner.load(str(a.checkpoint.resolve()),load_optimizer=False);policy=runner.get_inference_policy(device=env.device);obs,_=w.reset();rows=[]
 with torch.inference_mode():
  for tick in range(400):
   action=policy(obs);b=env.shuttle.data.root_pos_w[0]-env.scene.env_origins[0];v=env.shuttle.data.root_lin_vel_w[0];r=env.robot.data.body_pos_w[0,env.racket_body_id]-env.scene.env_origins[0];root=env.robot.data.root_pos_w[0]-env.scene.env_origins[0];rows.append({'time_s':tick*.01,'ball_xyz':b.cpu().tolist(),'ball_v':v.cpu().tolist(),'tool_xyz':r.cpu().tolist(),'root_xyz':root.cpu().tolist(),'contact':bool(env.contact_seen[0]),'crossed_net':bool(env.crossed_net[0]),'landed_target':bool(env.landed_target[0]),'airborne':bool(env.observed_airborne[0]),'action':action[0].cpu().tolist()});obs,_,_,_=w.step(action)
 (a.output/'trajectory.json').write_text(json.dumps(rows,indent=2));(a.output/'episode.json').write_text(json.dumps(env.ball_episode_records[-1] if env.ball_episode_records else {},indent=2));print(json.dumps(env.ball_episode_records[-1] if env.ball_episode_records else {},indent=2),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
