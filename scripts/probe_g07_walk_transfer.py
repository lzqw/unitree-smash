#!/usr/bin/env python3
"""Test a pinned pretrained walk actor under the original complete payload/net."""
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
p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--candidate',type=Path);p.add_argument('--fine',action='store_true')
p.add_argument('--manifest',type=Path,default=ROOT/'configs/g07_go2_walk_transfer.json')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv
from quadsmash.tasks.stand.net_contact_monitor import NetContactMonitor,add_net_contact_sensors

def main():
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=BallEnvCfg() if a.fine else JumpEnvCfg()
    if a.fine:
        cfg.sim.dt,cfg.decimation,cfg.sim.render_interval=.00025,40,40
        cfg.include_net=True
        cfg.shuttle_start=(12.,0.,.025)
        cfg.shuttle_velocity=(0.,0.,0.)
    cfg.scene.num_envs=a.num_envs
    cfg.seed=715001
    cfg.sim.device='cuda:0'
    cfg.episode_length_s=4.
    cfg.event_goal_enabled=True
    cfg.event_goal_position=(1.,0.,.3)
    candidate=torch.load(a.candidate,map_location='cpu',weights_only=False) if a.candidate else None
    target_x=candidate['result'].get('target_root_x_m',1.) if candidate else 1.
    cfg.event_goal_position=(target_x,0.,.3)
    class FineWalkEnv(NetContactMonitor,FullJointSmashEnv):
        def _setup_scene(self):
            super()._setup_scene()
            add_net_contact_sensors(self)
    env=FineWalkEnv(cfg) if a.fine else LoadedWalkEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    policy=Go2WalkTransfer(a.manifest,a.num_envs,env.device)
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    parameters=torch.tensor([[v,k,d] for v in [.35,.5,.7,1.]
        for k in [1.,2.,3.,4.] for d in [.2,.5,.8,1.2]],device=env.device)[:a.num_envs]
    if a.candidate:
        parameters=candidate['parameters'].to(env.device).repeat(a.num_envs,1)
    if len(parameters)!=a.num_envs:raise ValueError('Use no more than 64 lanes')
    obs,_=wrapped.reset()
    finished=torch.zeros(a.num_envs,dtype=torch.bool,device=env.device)
    final=torch.zeros((a.num_envs,3),device=env.device)
    velocity=final.clone()
    tilt=torch.zeros(a.num_envs,device=env.device)
    trace=[]
    joints=[]
    body_quaternions=[]
    try:
        with torch.inference_mode():
            for tick in range(400):
                active=~finished
                final[active]=(env.robot.data.root_pos_w-env.scene.env_origins)[active]
                velocity[active]=env.robot.data.root_link_lin_vel_w[active]
                tilt[active]=env.robot.data.projected_gravity_b[active,:2].norm(dim=1)
                if tick%2==0:
                    trace.append(final.cpu().numpy().copy())
                    joints.append(env.robot.data.joint_pos[:,env.joint_ids].cpu().numpy())
                    body_quaternions.append(env.robot.data.root_quat_w.cpu().numpy())
                obs,_,done,_=wrapped.step(policy.action(obs,parameters,default,limits))
                finished|=done.bool()
        first={}
        for row in env.episode_records:first.setdefault(row['env_id'],row)
        net={}
        for row in env.net_episode_records:net.setdefault(row['env_id'],row)
        if len(net)!=a.num_envs:raise RuntimeError('Incomplete net force telemetry')
        rows=[]
        for i in range(a.num_envs):
            r=first[i]
            valid=(not r['body_landing'] and not net[i]['robot_net_contact'] and abs(float(final[i,0])-target_x)<.06
                   and abs(float(final[i,1]))<.06 and float(final[i,2])>.2
                   and float(tilt[i])<.26 and float(velocity[i,:2].norm())<.15)
            rows.append(dict(env_id=i,parameters=parameters[i].cpu().tolist(),
                root_final_m=final[i].cpu().tolist(),velocity_mps=velocity[i].cpu().tolist(),
                tilt_gravity_xy=float(tilt[i]),movement_success=valid,jump_metrics=r,net_contact_telemetry=net[i]))
        best=max(rows,key=lambda r:(r['movement_success'],not r['jump_metrics']['body_landing'],
                 -abs(r['root_final_m'][0]-target_x)-abs(r['root_final_m'][1])))
        metrics=dict(kind='transferred learned walking probe, original full payload and net, no ball',
            target_root_x_m=target_x,net_contact_sensor_version='one-source-per-environment-v2',robot_net_contact=sum(r['robot_net_contact'] for r in net.values()),fine_physics_fixture=a.fine,candidate=str(a.candidate),physics_dt_s=cfg.sim.dt,formal_g07_acceptance=False,episodes=len(rows),movement_success=sum(r['movement_success'] for r in rows),
            source_manifest=json.loads(a.manifest.read_text()),best=best)
        (a.output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        np.save(a.output/'root_trajectories.npy',np.stack(trace))
        np.save(a.output/'joint_trajectories.npy',np.stack(joints))
        np.save(a.output/'root_quaternions.npy',np.stack(body_quaternions))
        (a.output/'completed.json').write_text(json.dumps(dict(episodes=len(rows),env_steps=400*a.num_envs))+'\n')
        print(json.dumps(metrics),flush=True)
    finally:env.close()

try:main()
except BaseException:
    a.output.mkdir(parents=True,exist_ok=True)
    error=traceback.format_exc()
    (a.output/'error.txt').write_text(error)
    print(error,file=sys.stderr,flush=True)
    raise
finally:app.close(skip_cleanup=True)
