#!/usr/bin/env python3
"""Fit new full-racket whole-body preparation/swing/recovery by native CEM."""
import argparse,hashlib,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--generations',type=int,default=8);p.add_argument('--seed',type=int,default=720501)
p.add_argument('--physics-dt',type=float,default=.001);p.add_argument('--resume',type=Path)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_reach_env_v4 import reach_cfg,FullRacketReachEnvV4
from quadsmash.control.full_racket_teacher_v4 import FullRacketTeacherV4,LOW,HIGH,INITIAL
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.physics.native import quat_rotate
from quadsmash.physics.full_racket_v4 import load_racket

stopping=False
def stop(*_):
    global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def seed_with_pitch_compensation():
    seed=json.loads((ROOT/'reports/G07_full_racket_v4/ik_seeds01/seeds.json').read_text())['seeds'][0].copy()
    model=FullRacketGeometryV4();q=np.array(seed['joint_position_rad'])
    point,rotation=model.forward(q,base_position=(0,0,0))
    point+=.0256*rotation[:,0]
    jacobian=[]
    for axis in np.eye(6):
        hi,rhi=model.forward(q+axis*1e-6,base_position=(0,0,0))
        lo,rlo=model.forward(q-axis*1e-6,base_position=(0,0,0))
        jacobian.append(np.r_[(hi+.0256*rhi[:,0]-lo-.0256*rlo[:,0])/2e-6,
                              .3*(rhi[:,0]-rlo[:,0])/2e-6])
    jacobian=np.stack(jacobian,axis=1)
    desired=np.r_[-np.cross([0,1,0],point),-.3*np.cross([0,1,0],rotation[:,0])]
    basis=np.linalg.lstsq(np.vstack([jacobian,.02*np.eye(6)]),np.r_[desired,np.zeros(6)],rcond=None)[0]
    seed['pitch_compensation_basis_rad_per_rad']=basis.tolist()
    return seed


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=json.loads(protocol_path.read_text())
    asset=ROOT/load_racket()['output_asset']
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==protocol['asset']['sha256']
    cfg=reach_cfg(args.num_envs,args.physics_dt);cfg.seed=args.seed
    env=FullRacketReachEnvV4(cfg);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    backbone_path=ROOT/'reports/G07/formal_candidate_v1/policy.pt'
    source=torch.load(backbone_path,map_location='cpu',weights_only=False)
    backbone={k:source[k] for k in ['walk_parameters','strike_policy_checkpoint','recovery_parameters']}
    seed=seed_with_pitch_compensation()
    teacher=FullRacketTeacherV4(backbone,seed,ROOT/'configs/g07_go2_walk_transfer.json',args.num_envs,env.device)
    low,high,mean=[torch.tensor(v,device=env.device) for v in [LOW,HIGH,INITIAL]]
    sigma=(high-low)*.18
    generator=torch.Generator(device=env.device).manual_seed(args.seed)
    history=[];all_best=-float('inf');previous_generations=0
    if args.resume:
        prior=torch.load(args.resume,map_location=env.device,weights_only=False)
        assert prior['schema']=='full-racket-reach-cem-v4'
        assert prior['asset_sha256']==protocol['asset']['sha256']
        mean=prior['mean'].clone();sigma=prior['sigma'].clone()
        generator.set_state(prior['rng'].cpu());previous_generations=prior['generations']
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    desired=torch.tensor([1.10,0,1.80],device=env.device)
    normal_goal=torch.tensor([.99503719,0,-.09950372],device=env.device)
    velocity_goal=torch.tensor([1.5,0,-.3],device=env.device)
    ticks=round(cfg.episode_length_s/.01)
    try:
        for generation in range(args.generations):
            parameters=(mean[None]+sigma[None]*torch.randn((args.num_envs,len(low)),generator=generator,device=env.device)).clamp(low,high)
            parameters[0]=mean
            if generation==0:parameters[1]=torch.tensor(INITIAL,device=env.device)
            env._episode_has_stepped.zero_();env._metrics_started=False
            env.episode_records.clear();env.reach_records.clear();teacher.reset()
            observation,_=wrapper.reset()
            finished=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
            best=torch.full((args.num_envs,),-100.,device=env.device)
            positions=torch.zeros((args.num_envs,3),device=env.device)
            velocities=torch.zeros_like(positions);normals=torch.zeros_like(positions);roots=torch.zeros_like(positions)
            event_time=torch.zeros(args.num_envs,device=env.device);air=finished.clone()
            with torch.inference_mode():
                for _ in range(ticks):
                    obs=observation['policy'];clock=obs[:,775];data=env.robot.data
                    center=data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins
                    normal=quat_rotate(data.body_link_quat_w[:,env.racket_body_id],torch.tensor([1.,0,0],device=env.device).expand(args.num_envs,-1))
                    point=center+.0256*normal
                    velocity=data.body_link_lin_vel_w[:,env.racket_body_id]+torch.linalg.cross(data.body_link_ang_vel_w[:,env.racket_body_id],.0256*normal)
                    score=-8*(point-desired).norm(dim=-1)-1.5*(normal-normal_goal).norm(dim=-1)-.30*(velocity-velocity_goal).norm(dim=-1)-3*(clock-2.75).abs()
                    score+=2*env.observed_airborne.float()-10*env.body_ground_seen.float()-10*env.net_seen.float()-10*env.body_crossed.float()
                    improved=(~finished)&(clock>=2.40)&(clock<=3.10)&(score>best)
                    best[improved]=score[improved];positions[improved]=point[improved];velocities[improved]=velocity[improved]
                    normals[improved]=normal[improved];event_time[improved]=clock[improved];air[improved]=env.observed_airborne[improved]
                    roots[improved]=(data.root_pos_w-env.scene.env_origins)[improved]
                    action=teacher.action(observation,parameters,default,limits)
                    observation,_,done,_=wrapper.step(action);finished|=done.bool()
            jumps,geometry={},{}
            for row in env.episode_records:jumps.setdefault(row['env_id'],row)
            for row in env.reach_records:geometry.setdefault(row['env_id'],row)
            assert len(jumps)==len(geometry)==args.num_envs,'Every candidate needs its first actual episode'
            rows=[]
            for i in range(args.num_envs):
                recovery=bool(jumps[i]['success']);safe=not geometry[i]['robot_net_contact'] and not geometry[i]['engineering_body_crossing']
                score=float(best[i])+3*recovery+min(jumps[i]['recovery_s'],.8)-10*(not safe)
                rows.append(dict(geometry[i],score=score,event_time_s=float(event_time[i]),
                    virtual_ball_center_m=positions[i].cpu().tolist(),point_velocity_mps=velocities[i].cpu().tolist(),
                    normal_world=normals[i].cpu().tolist(),root_at_event_m=roots[i].cpu().tolist(),
                    feet_airborne=bool(air[i]),recovery=recovery,jump=jumps[i]))
            scores=torch.tensor([r['score'] for r in rows],device=env.device)
            elite=scores.topk(max(2,args.num_envs//8)).indices
            mean=.25*mean+.75*parameters[elite].mean(0)
            sigma=(.25*sigma+.75*parameters[elite].std(0)).clamp_min((high-low)*.035)
            top=int(scores.argmax());entry=dict(generation=generation+previous_generations,
                best=rows[top],airborne_lanes=sum(r['feet_airborne'] for r in rows),
                recovered_lanes=sum(r['recovery'] for r in rows),net_contact_lanes=sum(r['robot_net_contact'] for r in rows),
                body_crossing_lanes=sum(r['engineering_body_crossing'] for r in rows),formal_acceptance=False)
            history.append(entry)
            population=dict(schema='full-racket-reach-cem-v4',parameters=parameters.cpu(),scores=scores.cpu(),
                mean=mean.cpu(),sigma=sigma.cpu(),rng=generator.get_state().cpu(),generations=generation+previous_generations+1,
                backbone=backbone,seed=seed,rows=rows,physics_dt_s=cfg.sim.dt,
                protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),asset_sha256=protocol['asset']['sha256'],
                source_backbone_sha256=hashlib.sha256(backbone_path.read_bytes()).hexdigest(),formal_acceptance=False,
                method='CEM of18 preparation/walk/jump/arm/swing/recovery parameters using actual full-racket dynamics')
            torch.save(population,args.output/f'population_{generation:03d}.pt')
            torch.save(population,args.output/'resume.pt')
            if float(scores[top])>all_best:
                all_best=float(scores[top])
                torch.save(dict(population,parameters=parameters[top].cpu(),result=rows[top],
                    kind='Learned no-ball curriculum controller; no contact or formal policy pass'),args.output/'best_candidate.pt')
            (args.output/f'episodes_{generation:03d}.json').write_text(json.dumps(rows,indent=2)+'\n')
            (args.output/'metrics.json').write_text(json.dumps(dict(history=history,
                best=max(history,key=lambda e:e['best']['score'])['best'],
                env_steps=len(history)*args.num_envs*ticks,formal_acceptance=False),indent=2)+'\n')
            print(json.dumps(entry),flush=True)
            if stopping:break
        if len(history)==args.generations:
            (args.output/'completed.json').write_text(json.dumps(dict(env_steps=len(history)*args.num_envs*ticks))+'\n')
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
