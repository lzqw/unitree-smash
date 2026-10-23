#!/usr/bin/env python3
"""Search real-PhysX high-arm trajectories around the best G07 candidate."""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=128)
p.add_argument('--physics-dt', type=float, default=.00025)
p.add_argument('--seed', type=int, default=613041)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); app = AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv, BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action

P = torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.25,.25,.15,-.20], dtype=torch.float32)
Q0 = torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722])
QD = torch.tensor([0.,1.,-1.,0.,0.,0.])
Q4 = torch.tensor([-1.89106,1.20556,-2.65668,.86670,-.88788,-.27291])
Q1 = torch.tensor([1.87380,1.33749,-2.84978,.20563,-.97821,-.11784])
try:
    a.output.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device='cuda').manual_seed(a.seed)
    cfg = BallEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.device='cuda:0'
    cfg.sim.dt=a.physics_dt; cfg.decimation=round(.01/a.physics_dt); cfg.sim.render_interval=cfg.decimation
    cfg.include_net=True; cfg.episode_length_s=1.8; cfg.arm_action_scale=2.; cfg.event_goal_position=(.3,0.,1.25); cfg.event_goal_normal=(0.,0.,1.)
    env=BallEnv(cfg); w=RslRlVecEnvWrapper(env, clip_actions=None); params=P.to(env.device).repeat(env.num_envs,1)
    default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone()
    # Stay in the high-hit basin found by swing_random02, while varying the
    # physical shuttle launch and the late arm pose/switch time.
    base0=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977], device=env.device)
    late=base0[None,:].repeat(a.num_envs,1)
    late += .13*torch.randn((a.num_envs,6), generator=g, device=env.device)
    late[:,1]=late[:,1].clamp(0,3.14); late[:,2]=late[:,2].clamp(-2.967,0); late[:,4]=late[:,4].clamp(-1.22,1.22); late[:,5]=late[:,5].clamp(-2.094,2.094)
    vx=-(5.95+.70*torch.rand(a.num_envs,generator=g,device=env.device))
    vz=6.55+.65*torch.rand(a.num_envs,generator=g,device=env.device)
    start_x=7.4*torch.ones_like(vx)
    switch=1.12+.16*torch.rand(a.num_envs,generator=g,device=env.device)
    cfg.shuttle_velocity=(float(vx.mean()),0.,float(vz.mean()))
    # Per-env reset values are written after reset, because the config only
    # accepts one launch vector. This remains a native RigidObject state write.
    obs,_=w.reset(); ids=torch.arange(a.num_envs,device=env.device)
    st=env.shuttle.data.default_root_state.clone(); st[:,:3]=env.scene.env_origins+torch.stack((start_x,torch.zeros_like(vx),1.15*torch.ones_like(vx)),dim=1); st[:,7:10]=torch.stack((vx,torch.zeros_like(vx),vz),dim=1); env.shuttle.write_root_state_to_sim(st, env_ids=ids); env.previous_ball_position.copy_(st[:,:3]); env.previous_ball_velocity.copy_(st[:,7:10])
    pre=(Q0.to(env.device)-default[None,:])/2.; late_action=(late-default[None,:])/2.
    with torch.inference_mode():
        for _ in range(182):
            action=teacher_action(obs['policy'],params); t=obs['policy'][:,775]
            arm=torch.zeros((a.num_envs,6),device=env.device); active=(t>=.70)&(t<1.42)
            arm[active]=torch.where((t[active]<switch[active])[:,None],pre[None,:],late_action[active]); action[:,12:]=arm
            obs,_,_,_=w.step(action)
    rows=[]
    for i,r in enumerate(list(env.ball_episode_records)):
        rows.append({'env_id':i,'vx_mps':float(vx[i]),'vz_mps':float(vz[i]),'switch_s':float(switch[i]),'contact':bool(r['contact']),'airborne_contact':bool(r['airborne_contact']),'body_contact':bool(r['body_contact']),'outgoing_speed_mps':float(r['outgoing_speed_mps']),'outgoing_vz_mps':float(r['outgoing_vz_mps']),'crossed_net':bool(r['crossed_net']),'landed_target':bool(r['landed_target']),'recovery_success':bool(r['recovery_success']),'contact_time_s':float(r['contact_time_s']),'ball_xyz':r['ball_min_distance_ball_xyz'],'tool_xyz':r['ball_min_distance_tool_xyz']})
    (a.output/'variants.json').write_text(json.dumps({'seed':a.seed,'late_joint_position':late.cpu().tolist()},indent=2)+'\n')
    m={'physics_dt_s':a.physics_dt,'arm_action_scale':2.,'episodes':len(rows),'contact':sum(v['contact'] for v in rows),'attack':sum(v['contact'] and v['outgoing_speed_mps']>=8 and v['outgoing_vz_mps']<=-1 for v in rows),'max_speed':max(v['outgoing_speed_mps'] for v in rows),'min_vz':min(v['outgoing_vz_mps'] for v in rows),'variants':rows}
    (a.output/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    with (a.output/'episodes.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=sorted(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(json.dumps({'contact':m['contact'],'attack':m['attack'],'max_speed':m['max_speed'],'min_vz':m['min_vz']},sort_keys=True),flush=True)
finally:
    try: env.close()
    except: pass
    app.close(skip_cleanup=True)
