#!/usr/bin/env python3
"""Probe full native arm joint limits, separate from the small policy action band."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=8192);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from quadsmash.tasks.stand.virtual_env import VirtualHitEnv,VirtualHitEnvCfg
from isaaclab.utils.math import quat_apply

env=None
try:
 a.output.mkdir(parents=True,exist_ok=True)
 cfg=VirtualHitEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device='cuda:0';cfg.seed=613001
 env=VirtualHitEnv(cfg)
 ids=env.joint_ids[12:]
 lim=env.robot.data.joint_pos_limits[:,ids]
 lo,hi=lim[:,:,0],lim[:,:,1]
 g=torch.Generator(device=env.device).manual_seed(613002)
 q=env.robot.data.default_joint_pos.clone();q[:,ids]=lo+(hi-lo)*torch.rand((a.num_envs,6),generator=g,device=env.device)
 env.robot.write_joint_state_to_sim(q,torch.zeros_like(q));env.sim.forward()
 pos=env.robot.data.body_pos_w[:,env.racket_body_id]-env.scene.env_origins;quat=env.robot.data.body_quat_w[:,env.racket_body_id];normal=quat_apply(quat,torch.tensor([1.,0.,0.],device=env.device).expand(a.num_envs,-1))
 z=pos[:,2];idx=torch.topk(z,k=min(10,a.num_envs)).indices
 out={'samples':a.num_envs,'joint_names':[env.robot.joint_names[i] for i in ids],'joint_limits':lim[0].cpu().tolist(),'position_min':pos.min(0).values.cpu().tolist(),'position_max':pos.max(0).values.cpu().tolist(),'normal_z_min':float(normal[:,2].min()),'normal_z_max':float(normal[:,2].max()),'normal_mean':normal.mean(0).cpu().tolist(),'max_z_configurations':[{ 'racket_xyz':pos[i].cpu().tolist(),'normal':normal[i].cpu().tolist(),'arm_joint_position':q[i,ids].cpu().tolist()} for i in idx]}
 (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
finally:
 if env is not None: env.close()
 app.close(skip_cleanup=True)
