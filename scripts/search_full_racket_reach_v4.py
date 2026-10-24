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
p.add_argument('--warm-start',type=Path);p.add_argument('--sigma-fraction',type=float,default=.18)
p.add_argument('--record-first-lanes',type=int,default=0)
p.add_argument('--cartesian-gain',type=float);p.add_argument('--swing-half-window',type=float)
p.add_argument('--ik-seeds',type=Path);p.add_argument('--walk-speed',type=float);p.add_argument('--waypoint-x',type=float)
p.add_argument('--position-weight',type=float,default=8.);p.add_argument('--velocity-weight',type=float,default=.30)
p.add_argument('--walk-thigh-offset',type=float);p.add_argument('--walk-calf-offset',type=float)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_reach_env_v4 import reach_cfg,FullRacketReachEnvV4,CONTACT_CONTRACT
from quadsmash.control.full_racket_teacher_v4 import FullRacketTeacherV4,LOW,HIGH,INITIAL,PARAMETER_NAMES
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.physics.native import quat_rotate
from quadsmash.physics.full_racket_v4 import load_racket

stopping=False
def stop(*_):
    global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def atomic_checkpoint(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,temp);temp.replace(path)


def seed_with_pitch_compensation(path=None):
    seed=json.loads((path or ROOT/'reports/G07_full_racket_v4/ik_seeds01/seeds.json').read_text())['seeds'][0].copy()
    model=FullRacketGeometryV4();q=np.array(seed['joint_position_rad'])
    point,rotation=model.forward(q,base_position=(0,0,0))
    offset=np.r_[seed.get('face_sign',1)*.0256,seed.get('string_uv_m',[0.,0.])]
    point+=rotation@offset
    jacobian=[]
    for axis in np.eye(6):
        hi,rhi=model.forward(q+axis*1e-6,base_position=(0,0,0))
        lo,rlo=model.forward(q-axis*1e-6,base_position=(0,0,0))
        jacobian.append(np.r_[(hi+rhi@offset-lo-rlo@offset)/2e-6,
                              .3*(rhi[:,0]-rlo[:,0])/2e-6])
    jacobian=np.stack(jacobian,axis=1)
    desired=np.r_[-np.cross([0,1,0],point),-.3*np.cross([0,1,0],rotation[:,0])]
    basis=np.linalg.lstsq(np.vstack([jacobian,.02*np.eye(6)]),np.r_[desired,np.zeros(6)],rcond=None)[0]
    seed['pitch_compensation_basis_rad_per_rad']=basis.tolist()
    return seed


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume and args.warm_start:raise ValueError('Choose optimizer resume or a fresh warm start')
    if args.num_envs<4 or args.generations<1 or not 0<args.sigma_fraction<=.5:raise ValueError('Invalid search batch/settings')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=json.loads(protocol_path.read_text())
    asset=ROOT/load_racket()['output_asset']
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==protocol['asset']['sha256']
    cfg=reach_cfg(args.num_envs,args.physics_dt);cfg.seed=args.seed
    env=FullRacketReachEnvV4(cfg);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    backbone_path=ROOT/'reports/G07/formal_candidate_v1/policy.pt'
    source=torch.load(backbone_path,map_location='cpu',weights_only=False)
    backbone={k:source[k] for k in ['walk_parameters','strike_policy_checkpoint','recovery_parameters']}
    seed=seed_with_pitch_compensation();warm=None;prior=None
    if args.warm_start:
        warm=torch.load(args.warm_start,map_location='cpu',weights_only=False)
        assert warm['schema']=='full-racket-reach-cem-v4' and warm['parameters'].ndim==1
        assert warm['asset_sha256']==protocol['asset']['sha256']
        backbone,seed=warm['backbone'],warm['seed']
    if args.resume:
        prior=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert prior['schema']=='full-racket-reach-cem-v4'
        assert prior['asset_sha256']==protocol['asset']['sha256']
        assert prior['physics_dt_s']==args.physics_dt,'Use warm-start when changing integration step; never inherit old fitness'
        assert prior.get('contact_contract')==CONTACT_CONTRACT,'Contact contract changed; use warm-start and remeasure all fitness'
        assert prior.get('search_objective')==dict(position_weight=args.position_weight,velocity_weight=args.velocity_weight),'Objective changed; use warm-start and remeasure fitness'
        assert not args.ik_seeds,'Changing the intercept seed requires a fresh warm start'
        backbone,seed=prior['backbone'],prior['seed']
    if args.ik_seeds:seed=seed_with_pitch_compensation(args.ik_seeds)
    teacher=FullRacketTeacherV4(backbone,seed,ROOT/'configs/g07_go2_walk_transfer.json',args.num_envs,env.device)
    low,high,mean=[torch.tensor(v,device=env.device) for v in [LOW,HIGH,INITIAL]]
    sigma=(high-low)*args.sigma_fraction
    if warm is not None:
        seed_parameters=warm['parameters'].to(env.device)
        if len(seed_parameters)==18:seed_parameters=torch.cat((seed_parameters,torch.tensor([0.,.12],device=env.device)))
        if len(seed_parameters)<len(low):seed_parameters=torch.cat((seed_parameters,torch.zeros(len(low)-len(seed_parameters),device=env.device)))
        mean=seed_parameters.clamp(low,high)
    reference=mean.clone()
    if args.cartesian_gain is not None:mean[18]=args.cartesian_gain
    if args.swing_half_window is not None:mean[19]=args.swing_half_window
    if args.walk_speed is not None:mean[0]=args.walk_speed
    if args.waypoint_x is not None:mean[1]=args.waypoint_x
    if args.ik_seeds:mean[8:14]=0.;mean[7]=1.
    if args.walk_thigh_offset is not None:mean[20]=args.walk_thigh_offset
    if args.walk_calf_offset is not None:mean[21]=args.walk_calf_offset
    mean=mean.clamp(low,high)
    if args.ik_seeds:
        reference=mean.clone();reference[18]=0.
    generator=torch.Generator(device=env.device).manual_seed(args.seed)
    history=[];all_best=-float('inf');all_best_rank=(-1,-float('inf'));previous_generations=0;best_parameters=None;best_result=None
    if args.resume:
        mean=prior['mean'].to(env.device).clone();sigma=prior['sigma'].to(env.device).clone()
        assert len(mean)==len(low),'Parameterization changed; use warm-start to retain the old controller as a reference'
        assert prior.get('selection_contract')=='safe-virtual-own-side-v4','Selection contract changed; use warm-start and remeasure'
        reference=prior.get('reference_parameters',torch.tensor(INITIAL)).to(env.device)
        generator.set_state(prior['rng'].cpu());previous_generations=prior['generations']
        if prior.get('best_parameters') is not None:
            best_parameters=prior['best_parameters'].to(env.device);best_result=prior['best_result'];all_best=best_result['score']
            all_best_rank=(int(best_result.get('virtual_own_side_safe',False)),all_best)
    limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    default=env.robot.data.default_joint_pos[0,env.joint_ids]
    desired=torch.tensor(seed.get('desired_ball_center_m',[1.10,0,1.80]),device=env.device)
    normal_goal=torch.tensor(seed.get('outward_contact_normal_world',[.99503719,0,-.09950372]),device=env.device)
    velocity_goal=torch.tensor(seed.get('desired_point_velocity_mps',[1.5,0,-.3]),device=env.device)
    face_sign=seed.get('face_sign',1)
    contact_offset=torch.tensor([face_sign*.0256,*seed.get('string_uv_m',[0.,0.])],device=env.device).expand(args.num_envs,-1)
    ticks=round(cfg.episode_length_s/.01)
    recorded=min(args.record_first_lanes,args.num_envs)
    if recorded:
        from isaacsim.core.utils.stage import get_current_stage
        from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
        export_robot_geometry(get_current_stage(),env.robot,args.output)
    try:
        for generation in range(args.generations):
            # This checkpoint replays an interrupted generation from the
            # ground. It does not claim to serialize hidden PhysX caches.
            atomic_checkpoint(args.output/'generation_start.pt',dict(
                schema='full-racket-reach-cem-v4',mean=mean.cpu(),sigma=sigma.cpu(),
                rng=generator.get_state().cpu(),generations=generation+previous_generations,
                backbone=backbone,seed=seed,physics_dt_s=cfg.sim.dt,
                contact_contract=CONTACT_CONTRACT,
                reference_parameters=reference.cpu(),
                selection_contract='safe-virtual-own-side-v4',
                asset_sha256=protocol['asset']['sha256'],
                best_parameters=best_parameters.cpu() if best_parameters is not None else None,best_result=best_result))
            parameters=(mean[None]+sigma[None]*torch.randn((args.num_envs,len(low)),generator=generator,device=env.device)).clamp(low,high)
            parameters[0]=mean
            if best_parameters is not None:parameters[1]=best_parameters
            elif generation==0:parameters[1]=reference
            env._episode_has_stepped.zero_();env._metrics_started=False
            env.episode_records.clear();env.reach_records.clear();teacher.reset()
            observation,_=wrapper.reset()
            finished=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
            best=torch.full((args.num_envs,),-100.,device=env.device)
            positions=torch.zeros((args.num_envs,3),device=env.device)
            velocities=torch.zeros_like(positions);normals=torch.zeros_like(positions);roots=torch.zeros_like(positions)
            event_time=torch.zeros(args.num_envs,device=env.device);air=finished.clone()
            lifetime=torch.zeros(args.num_envs,device=env.device)
            minimum_root=torch.full((args.num_envs,),100.,device=env.device)
            prefix_tilt=torch.zeros(args.num_envs,device=env.device)
            prepared=finished.clone()
            public_trace,action_trace,pose_trace,valid_trace=[],[],[],[]
            with torch.inference_mode():
                for tick in range(ticks):
                    obs=observation['policy'];clock=obs[:,775];data=env.robot.data
                    lifetime=torch.where(~finished,clock,lifetime)
                    minimum_root=torch.where(~finished,torch.minimum(minimum_root,data.root_pos_w[:,2]),minimum_root)
                    prefix_tilt=torch.where((~finished)&(clock<2.),torch.maximum(prefix_tilt,data.projected_gravity_b[:,:2].norm(dim=-1)),prefix_tilt)
                    prepared|=(~finished)&(clock>=2.)
                    center=data.body_link_pos_w[:,env.racket_body_id]-env.scene.env_origins
                    normal=face_sign*quat_rotate(data.body_link_quat_w[:,env.racket_body_id],torch.tensor([1.,0,0],device=env.device).expand(args.num_envs,-1))
                    displacement=quat_rotate(data.body_link_quat_w[:,env.racket_body_id],contact_offset)
                    point=center+displacement
                    velocity=data.body_link_lin_vel_w[:,env.racket_body_id]+torch.linalg.cross(data.body_link_ang_vel_w[:,env.racket_body_id],displacement)
                    score=-args.position_weight*(point-desired).norm(dim=-1)-1.5*(normal-normal_goal).norm(dim=-1)-args.velocity_weight*(velocity-velocity_goal).norm(dim=-1)-3*(clock-2.75).abs()
                    score+=2*env.observed_airborne.float()-10*env.body_ground_seen.float()-10*env.net_seen.float()-10*env.body_crossed.float()
                    score-=30*(point[:,0]>=1.5).float()
                    improved=(~finished)&(clock>=2.40)&(clock<=3.10)&(score>best)
                    best[improved]=score[improved];positions[improved]=point[improved];velocities[improved]=velocity[improved]
                    normals[improved]=normal[improved];event_time[improved]=clock[improved];air[improved]=env.observed_airborne[improved]
                    roots[improved]=(data.root_pos_w-env.scene.env_origins)[improved]
                    action=teacher.action(observation,parameters,default,limits)
                    if recorded:
                        public_trace.append(obs[:recorded].cpu().numpy().copy())
                        action_trace.append(action[:recorded].cpu().numpy().copy())
                        pose_trace.append(torch.cat((data.body_link_pos_w[:recorded]-env.scene.env_origins[:recorded,None],data.body_link_quat_w[:recorded]),-1).cpu().numpy())
                        valid_trace.append((~finished[:recorded]).cpu().numpy().copy())
                    observation,_,done,_=wrapper.step(action);finished|=done.bool()
                    if (tick+1)%100==0:
                        partial=dict(generation=generation+previous_generations,control_ticks=tick+1,requested_ticks=ticks,
                            first_episodes_terminated=int(finished.sum()),preparation_survived=int(prepared.sum()),
                            physics_dt_s=args.physics_dt,formal_acceptance=False)
                        temp=args.output/'progress.json.tmp';temp.write_text(json.dumps(partial,indent=2)+'\n');temp.replace(args.output/'progress.json')
                        print(json.dumps(partial),flush=True)
                    if stopping and tick+1<ticks:
                        (args.output/'interrupted.json').write_text(json.dumps(dict(
                            generation=generation+previous_generations,control_ticks=tick+1,
                            resume_from='generation_start.pt',resume_semantics='Replay this generation from ground; no mid-PhysX-state restore',
                            completed_generations=len(history),formal_acceptance=False),indent=2)+'\n')
                        if recorded:
                            np.savez_compressed(args.output/'interrupted_public_and_poses.npz',
                                observations=np.stack(public_trace),actions=np.stack(action_trace),link_poses=np.stack(pose_trace),valid_first_episode=np.stack(valid_trace))
                        break
            if stopping and tick+1<ticks:break
            jumps,geometry={},{}
            for row in env.episode_records:jumps.setdefault(row['env_id'],row)
            for row in env.reach_records:geometry.setdefault(row['env_id'],row)
            assert len(jumps)==len(geometry)==args.num_envs,'Every candidate needs its first actual episode'
            rows=[]
            for i in range(args.num_envs):
                recovery=bool(jumps[i]['success']);safe=not geometry[i]['robot_net_contact'] and not geometry[i]['engineering_body_crossing']
                score=float(best[i])+3*recovery+min(jumps[i]['recovery_s'],.8)-10*(not safe)
                if float(best[i])<=-99.:score=-100.+5*float(lifetime[i])-10*(not safe)
                rows.append(dict(geometry[i],score=score,event_time_s=float(event_time[i]),
                    virtual_ball_center_m=positions[i].cpu().tolist(),point_velocity_mps=velocities[i].cpu().tolist(),
                    normal_world=normals[i].cpu().tolist(),root_at_event_m=roots[i].cpu().tolist(),
                    feet_airborne=bool(air[i]),recovery=recovery,jump=jumps[i],
                    preparation_survived_to2s=bool(prepared[i]),minimum_root_height_m=float(minimum_root[i]),
                    preparation_maximum_gravity_xy=float(prefix_tilt[i]),first_episode_lifetime_s=float(lifetime[i])))
                rows[-1]['virtual_own_side_safe']=bool(safe and best[i]>-99 and positions[i,0]<1.5)
            scores=torch.tensor([r['score'] for r in rows],device=env.device)
            feasible=torch.tensor([r['virtual_own_side_safe'] for r in rows],device=env.device)
            feasible_count=int(feasible.sum())
            selection_scores=torch.where(feasible,scores,scores-1000.) if feasible_count else scores
            elite=selection_scores.topk(min(feasible_count,max(2,args.num_envs//8)) if feasible_count else max(2,args.num_envs//8)).indices
            mean=.25*mean+.75*parameters[elite].mean(0)
            sigma=(.25*sigma+.75*parameters[elite].std(0,unbiased=False)).clamp_min((high-low)*.035)
            top=int(selection_scores.argmax());entry=dict(generation=generation+previous_generations,
                best=rows[top],airborne_lanes=sum(r['feet_airborne'] for r in rows),
                recovered_lanes=sum(r['recovery'] for r in rows),net_contact_lanes=sum(r['robot_net_contact'] for r in rows),
                body_crossing_lanes=sum(r['engineering_body_crossing'] for r in rows),virtual_own_side_safe_lanes=feasible_count,formal_acceptance=False)
            history.append(entry)
            rank=(int(feasible[top]),float(scores[top]))
            if rank>all_best_rank:
                all_best_rank=rank
                all_best=float(scores[top]);best_parameters=parameters[top].clone();best_result=rows[top]
            population=dict(schema='full-racket-reach-cem-v4',parameters=parameters.cpu(),scores=scores.cpu(),
                mean=mean.cpu(),sigma=sigma.cpu(),rng=generator.get_state().cpu(),generations=generation+previous_generations+1,
                backbone=backbone,seed=seed,rows=rows,physics_dt_s=cfg.sim.dt,
                contact_contract=CONTACT_CONTRACT,
                protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),asset_sha256=protocol['asset']['sha256'],
                source_backbone_sha256=hashlib.sha256(backbone_path.read_bytes()).hexdigest(),formal_acceptance=False,
                best_parameters=best_parameters.cpu(),best_result=best_result,
                reference_parameters=reference.cpu(),
                selection_contract='safe-virtual-own-side-v4',
                search_objective=dict(position_weight=args.position_weight,velocity_weight=args.velocity_weight),
                warm_start_source=str(args.warm_start) if args.warm_start else None,
                parameter_names=PARAMETER_NAMES,
                method=f'CEM of{len(low)} preparation/walk/jump/arm/swing/recovery parameters using actual full-racket dynamics')
            torch.save(population,args.output/f'population_{generation:03d}.pt')
            atomic_checkpoint(args.output/'resume.pt',population)
            atomic_checkpoint(args.output/'best_candidate.pt',dict(population,parameters=best_parameters.cpu(),result=best_result,
                kind='Learned no-ball curriculum controller; no contact or formal policy pass'))
            if recorded:
                np.savez_compressed(args.output/f'public_and_poses_{generation:03d}.npz',
                    observations=np.stack(public_trace),actions=np.stack(action_trace),link_poses=np.stack(pose_trace),valid_first_episode=np.stack(valid_trace))
            (args.output/f'episodes_{generation:03d}.json').write_text(json.dumps(rows,indent=2)+'\n')
            (args.output/'metrics.json').write_text(json.dumps(dict(history=history,
                best=best_result,
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
