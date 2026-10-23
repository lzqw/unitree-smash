#!/usr/bin/env python3
"""Positive-control diagnostic for the separate robot/net contact sensors."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnvCfg
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv
from quadsmash.control.piecewise_affine import PiecewiseAffinePolicy

def main():
 a.output.mkdir(parents=True,exist_ok=True);cfg=JumpEnvCfg();cfg.scene.num_envs=4;cfg.seed=715701;cfg.sim.device='cuda:0';cfg.robot.init_state.pos=(1.,0.,.323);cfg.episode_length_s=.8;env=LoadedWalkEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);policy=PiecewiseAffinePolicy(torch.load(ROOT/'reports/G07/fit_piecewise_curriculum03/policy.pt',map_location='cpu',weights_only=False)).to(env.device);obs,_=w.reset();trace=[]
 try:
  with torch.inference_mode():
   for tick in range(80):
    trace.append(dict(time_s=tick*.01,net_force_max_n=env.net_robot_max_force.cpu().tolist(),tool_total_force_n=env.scene.sensors[env._net_racket_sensor_name].data.net_forces_w.norm(dim=-1).sum(1).cpu().tolist()));obs,_,_,_=w.step(policy(obs['policy']))
  rows=list(env.net_episode_records);result=dict(kind='intentional old-policy net interference diagnostic; not a valid motion',episodes=len(rows),detected=sum(r['robot_net_contact'] for r in rows),rows=rows);(a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'trace.json').write_text(json.dumps(trace)+'\n');(a.output/'completed.json').write_text(json.dumps(dict(episodes=len(rows)))+'\n');print(json.dumps(result),flush=True)
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
