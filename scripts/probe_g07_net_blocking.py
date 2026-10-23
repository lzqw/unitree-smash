#!/usr/bin/env python3
"""Targeted native check: every cloned net must block a below-top shuttle."""
import argparse
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=32)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import torch
import omni.usd
from pxr import Usd,UsdPhysics,UsdGeom
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv

def main():
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=BallEnvCfg()
    cfg.scene.num_envs=a.num_envs
    cfg.seed=715401
    cfg.sim.device='cuda:0'
    cfg.sim.dt,cfg.decimation,cfg.sim.render_interval=.00025,40,40
    cfg.include_net=True
    cfg.shuttle_start=(1.,0.,1.)
    cfg.shuttle_velocity=(5.,0.,0.)
    cfg.episode_length_s=1.
    env=FullJointSmashEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    default=env.robot.data.default_joint_pos[:,env.joint_ids]
    limits=env.robot.data.joint_pos_limits[:,env.joint_ids]
    action=(2*(default-limits[:,:,0])/(limits[:,:,1]-limits[:,:,0])-1).clamp(-1,1)
    wrapped.reset()
    trace=[]
    stage=omni.usd.get_context().get_stage()
    geometry=[]
    for i in range(a.num_envs):
        root=stage.GetPrimAtPath(f'/World/envs/env_{i}/Net')
        colliders=[]
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(dict(path=str(prim.GetPath()),enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()))
        transform=UsdGeom.XformCache().GetLocalToWorldTransform(root)
        center=list(transform.ExtractTranslation())
        origin=env.scene.env_origins[i].cpu().tolist()
        geometry.append(dict(env_id=i,exists=root.IsValid(),colliders=colliders,
            net_center_relative_to_environment_m=[v-o for v,o in zip(center,origin)],
            expected_center_relative_m=[1.5,0.,.762]))
    try:
        with torch.inference_mode():
            for tick in range(50):
                position=env.shuttle.data.root_pos_w-env.scene.env_origins
                trace.append(dict(time_s=tick*.01,position=position.cpu().tolist(),velocity=env.shuttle.data.root_lin_vel_w.cpu().tolist()))
                wrapped.step(action)
        maximum=torch.tensor([t['position'] for t in trace],device=env.device)[:,:,0].max(0).values
        final=env.shuttle.data.root_pos_w-env.scene.env_origins
        velocity=env.shuttle.data.root_lin_vel_w
        # Later rebounds from the robot/ground can change final vx again.
        # Verify the first net-side reversal and absence of a through-crossing.
        sampled_position=torch.tensor([t['position'] for t in trace],device=env.device)
        sampled_velocity=torch.tensor([t['velocity'] for t in trace],device=env.device)
        reflected=((sampled_position[:,:,0]>1.4)&(sampled_velocity[:,:,0]<-.1)).any(0)
        blocked=(maximum<1.535)&reflected
        result=dict(kind='native below-top net collision diagnostic',num_envs=a.num_envs,
            blocked=int(blocked.sum()),all_nets_block=bool(blocked.all()),
            lanes=[dict(env_id=i,blocked=bool(blocked[i]),maximum_x_m=float(maximum[i]),
                final_position_m=final[i].cpu().tolist(),final_velocity_mps=velocity[i].cpu().tolist()) for i in range(a.num_envs)],
            geometry=geometry,env_steps=50*a.num_envs)
        (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        (a.output/'trace.json').write_text(json.dumps(trace)+'\n')
        (a.output/'completed.json').write_text(json.dumps(dict(checked=a.num_envs,all_nets_block=bool(blocked.all())))+'\n')
        print(json.dumps(result),flush=True)
    finally:env.close()

try:main()
except BaseException:
    a.output.mkdir(parents=True,exist_ok=True)
    error=traceback.format_exc()
    (a.output/'error.txt').write_text(error)
    print(error,file=sys.stderr,flush=True)
    raise
finally:app.close(skip_cleanup=True)
