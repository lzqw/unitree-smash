#!/usr/bin/env python3
"""Adapt arm posture and position commands around a frozen learned walk actor."""
import argparse
import json
import signal
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--generations',type=int,default=30)
p.add_argument('--initial',type=Path)
p.add_argument('--wrist',action='store_true')
p.add_argument('--high-arm',action='store_true')
p.add_argument('--arm-seed',type=Path)
p.add_argument('--target-x',type=float,default=1.)
p.add_argument('--manifest',type=Path,default=ROOT/'configs/g07_go2_walk_transfer.json')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnvCfg
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.physics.native import quat_rotate
stopping=False
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)

def main():
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=JumpEnvCfg()
    cfg.scene.num_envs=a.num_envs
    cfg.seed=715101
    cfg.sim.device='cuda:0'
    cfg.sim.dt,cfg.decimation,cfg.sim.render_interval=.002,5,5
    cfg.episode_length_s=4.
    cfg.event_goal_enabled=True
    cfg.event_goal_position=(a.target_x,0.,.3)
    env=LoadedWalkEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    policy=Go2WalkTransfer(a.manifest,a.num_envs,env.device)
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    low=torch.tensor([.25,.5,0.,-.4,-.25,-.3,-.6,.6,.7,-2.7,.15],device=env.device)
    high=torch.tensor([1.,4.,1.2,.4,.25,.3,.6,1.3,1.8,-1.,.8],device=env.device)
    mean=torch.tensor([.65,2.,.5,-.10,0.,0.,.1,1.,1.5,-2.4,.5],device=env.device)
    sigma=(high-low)*.18
    if a.wrist:
        low=torch.cat((low,torch.tensor([-1.22],device=env.device)))
        high=torch.cat((high,torch.tensor([1.22],device=env.device)))
        mean=torch.cat((mean,torch.tensor([1.10],device=env.device)))
        sigma=(high-low)*.15
    if a.initial:
        prior=torch.load(a.initial,map_location=env.device,weights_only=False)
        seed_count=min(len(prior['parameters']),len(mean))
        mean[:seed_count]=prior['parameters'][:seed_count]
        sigma=(high-low)*.05
        if a.wrist:sigma[-1]=.25
    if a.high_arm:
        if not a.arm_seed:raise ValueError('high-arm requires an arm pose seed')
        arm=torch.load(a.arm_seed,map_location=env.device,weights_only=False)['arm_pose']
        delay=(prior['parameters'][14].clone() if a.initial and len(prior['parameters'])==15 else mean[10].clone())
        low=torch.cat((low[:8],torch.maximum(limits[12:,0],arm-.35),torch.tensor([.15],device=env.device)))
        high=torch.cat((high[:8],torch.minimum(limits[12:,1],arm+.35),torch.tensor([.8],device=env.device)))
        mean=torch.cat((mean[:8],arm,delay[None])).clamp(low,high)
        if a.initial and len(prior['parameters'])==15:mean=prior['parameters'].clamp(low,high)
        sigma=(high-low)*.10
    generator=torch.Generator(device=env.device).manual_seed(cfg.seed)
    history=[]
    best=-float('inf')
    try:
        for generation in range(a.generations):
            params=(mean[None,:]+sigma[None,:]*torch.randn((a.num_envs,len(low)),generator=generator,device=env.device)).clamp(low,high)
            params[0]=mean
            count=a.num_envs//5
            params[-count:]=low+(high-low)*torch.rand((count,len(low)),generator=generator,device=env.device)
            env._episode_has_stepped.zero_()
            env._metrics_started=False
            env.episode_records.clear()
            env.net_episode_records.clear()
            policy.reset()
            obs,_=wrapped.reset()
            finished=torch.zeros(a.num_envs,dtype=torch.bool,device=env.device)
            final=torch.zeros((a.num_envs,3),device=env.device)
            velocity=final.clone()
            tilt=torch.zeros(a.num_envs,device=env.device)
            interference=torch.zeros_like(finished)
            with torch.inference_mode():
                for _ in range(400):
                    active=~finished
                    final[active]=(env.robot.data.root_pos_w-env.scene.env_origins)[active]
                    velocity[active]=env.robot.data.root_link_lin_vel_w[active]
                    tilt[active]=env.robot.data.projected_gravity_b[active,:2].norm(dim=1)
                    tool=env.robot.data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins
                    quat=env.robot.data.body_link_quat_w[:,env.racket_body_id]
                    axes=quat_rotate(quat[:,None,:].expand(-1,3,-1),torch.eye(3,device=env.device)[None].expand(a.num_envs,-1,-1))
                    support=(axes.abs()*torch.tensor([.0075,.11,.11],device=env.device)[None,:,None]).sum(1)
                    interference|=active&((tool[:,0]+support[:,0])>1.46)&((tool[:,2]-support[:,2])<1.554)
                    obs,_,done,_=wrapped.step(policy.action(obs,params,default,limits))
                    finished|=done.bool()
            first={}
            for row in env.episode_records:first.setdefault(row['env_id'],row)
            first_net={}
            for row in env.net_episode_records:first_net.setdefault(row['env_id'],row)
            if len(first_net)!=a.num_envs:raise RuntimeError('Incomplete net force telemetry')
            net_bad=torch.tensor([first_net[i]['robot_net_contact'] for i in range(a.num_envs)],device=env.device)
            safe=torch.tensor([not first[i]['body_landing'] for i in range(a.num_envs)],device=env.device)
            upright=safe&(~interference)&(~net_bad)&(final[:,2]>.2)&(tilt<.26)
            success=upright&((final[:,0]-a.target_x).abs()<.06)&(final[:,1].abs()<.06)&(velocity[:,:2].norm(dim=1)<.15)
            scores=50*upright.float()+10*success.float()-8*(final[:,0]-a.target_x).abs()-4*final[:,1].abs()-2*tilt-velocity.norm(dim=1)
            elite=scores.topk(max(8,a.num_envs//8)).indices
            mean=.25*mean+.75*params[elite].mean(0)
            sigma=(.25*sigma+.75*params[elite].std(0)).clamp(min=(high-low)*.02)
            i=int(scores.argmax())
            result=dict(generation=generation,score=float(scores[i]),parameters=params[i].cpu().tolist(),
                root_final_m=final[i].cpu().tolist(),velocity_mps=velocity[i].cpu().tolist(),
                upright=bool(upright[i]),movement_success=bool(success[i]),successes=int(success.sum()),
                upright_lanes=int(upright.sum()),jump_metrics=first[i],
                potential_net_interference=bool(interference[i]),net_geometry_version=env.net_geometry_version,
                robot_net_contact=bool(net_bad[i]),net_contact_telemetry=first_net[i],
                target_root_x_m=a.target_x,parameter_schema='walk-high-arm-15' if a.high_arm else 'walk-offsets-12',
                kind='frozen walk actor plus searched arm posture/feedback; original payload/net; no ball')
            if float(scores[i])>best:
                best=float(scores[i])
                torch.save(dict(parameters=params[i].cpu(),result=result,manifest=json.loads(a.manifest.read_text())),a.output/'best_candidate.pt')
            history.append(result)
            (a.output/'metrics.json').write_text(json.dumps(dict(history=history,best=max(history,key=lambda r:r['score']),env_steps=(generation+1)*400*a.num_envs),indent=2)+'\n')
            torch.save(dict(parameters=params.cpu(),root_final=final.cpu(),velocity=velocity.cpu(),scores=scores.cpu(),mean=mean.cpu(),sigma=sigma.cpu(),rng=generator.get_state().cpu()),a.output/f'population_{generation:03d}.pt')
            print(json.dumps(result),flush=True)
            if stopping:break
        if len(history)==a.generations:(a.output/'completed.json').write_text(json.dumps(dict(generations=len(history)))+'\n')
    finally:env.close()

try:main()
except BaseException:
    a.output.mkdir(parents=True,exist_ok=True)
    error=traceback.format_exc()
    (a.output/'error.txt').write_text(error)
    print(error,file=sys.stderr,flush=True)
    raise
finally:app.close(skip_cleanup=True)
