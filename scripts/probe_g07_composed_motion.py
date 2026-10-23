#!/usr/bin/env python3
"""Replay a frozen canonical no-ball preparation to locate robot/net contacts."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnvCfg
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.control.composed_reach_teacher import composed_reach_action

def main():
 a.output.mkdir(parents=True,exist_ok=True);c=torch.load(a.candidate,map_location='cpu',weights_only=False);cfg=JumpEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=716001;cfg.sim.device='cuda:0';cfg.episode_length_s=c['walk_seconds']+4.;cfg.event_goal_enabled=True;cfg.event_goal_position=(c['walk_waypoint_x'],0.,.3);cfg.event_goal_time_s=c['result']['event_time_s'];env=LoadedWalkEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);d=env.device;walk=Go2WalkTransfer(ROOT/'configs/g07_go2_walk_transfer.json',a.num_envs,d);wp=c['walk_parameters'].to(d).repeat(a.num_envs,1);parameters=c['parameters'].to(d).repeat(a.num_envs,1);arm=c['arm_pose'].to(d);initial=c['arm_initial_pose'].to(d);default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids];obs,_=w.reset();trace=[];wt=round(c['walk_seconds']/.01);ticks=round(cfg.episode_length_s/.01)
 try:
  with torch.inference_mode():
   for tick in range(ticks):
    if tick<wt:action=walk.action(obs,wp,default,limits)
    else:
     if tick==wt:env.goal_position_w.copy_(env.scene.env_origins+torch.tensor(c['result']['virtual_ball_center_m'],device=d))
     local={'policy':obs['policy'].clone()};local['policy'][:,775]-=c['walk_seconds'];action=composed_reach_action(local,parameters,default,limits,arm,initial)
    if tick%2==0:trace.append(dict(time_s=tick*.01,root_xyz=(env.robot.data.root_pos_w-env.scene.env_origins).cpu().tolist(),net_max_force_n=env.net_robot_max_force.cpu().tolist()))
    obs,_,_,_=w.step(action)
  result=dict(kind='frozen canonical no-ball diagnostic, not a smash',candidate=str(a.candidate),net_episodes=env.net_episode_records,jump_episodes=env.episode_records);(a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'trace.json').write_text(json.dumps(trace)+'\n');(a.output/'completed.json').write_text(json.dumps(dict(env_steps=a.num_envs*ticks))+'\n');print(json.dumps(result),flush=True)
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
