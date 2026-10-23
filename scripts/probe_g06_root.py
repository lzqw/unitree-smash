#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/G06/policy.pt');p.add_argument('--physics-dt',type=float,default=.00025);p.add_argument('--leg-mult',type=float,default=1.0);p.add_argument('--leg-crouch',type=float,default=.25);p.add_argument('--leg-push',type=float,default=.55);p.add_argument('--num-envs',type=int,default=1);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
try:
 a.output.mkdir(parents=True,exist_ok=True);cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=1909;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.arm_action_scale=0.12;cfg.leg_action_scale_crouch=a.leg_crouch;cfg.leg_action_scale_push=a.leg_push;cfg.shuttle_start=(8.18,0.,1.15);cfg.shuttle_velocity=(-6.6,0.,6.82);cfg.event_goal_position=(.3,0.,1.25);cfg.event_goal_normal=(0.,0.,1.);cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=1.0);config=json.loads((ROOT/'configs/g05_imitation.json').read_text());runner=OnPolicyRunner(w,config['runner'],log_dir=str(a.output/'ppo'),device=env.device);runner.load(str(a.checkpoint.resolve()),load_optimizer=False);policy=runner.get_inference_policy(device=env.device);obs,_=w.reset();rows=[]
 with torch.inference_mode():
  for tick in range(400):
   action=policy(obs);mult=torch.full((a.num_envs,1),a.leg_mult,device=env.device);mult[:min(a.num_envs,4),0]=torch.tensor([1.,1.5,2.,3.],device=env.device)[:min(a.num_envs,4)];action[:,:12]=(mult*action[:,:12]).clamp(-1,1);root=env.robot.data.root_pos_w-env.scene.env_origins;tool=env.robot.data.body_pos_w[:,env.racket_body_id]-env.scene.env_origins;rows.append({'time_s':tick*.01,'root_xyz':root.cpu().tolist(),'tool_xyz':tool.cpu().tolist(),'airborne':env.observed_airborne.cpu().tolist(),'action_legs':action[:,:12].cpu().tolist()});obs,_,_,_=w.step(action)
 (a.output/'trajectory.json').write_text(json.dumps(rows,indent=2));print('max_root_z',max(max(v[2] for v in r['root_xyz']) for r in rows),'max_tool_z',max(max(v[2] for v in r['tool_xyz']) for r in rows),flush=True)
finally:
 try:env.close()
 except:pass
 app.close(skip_cleanup=True)
