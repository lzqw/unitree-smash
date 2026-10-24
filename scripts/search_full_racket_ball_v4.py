#!/usr/bin/env python3
"""Adapt whole-body preparation, interception and recovery on frozen TRAIN cases."""
import argparse,hashlib,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--warm-start',type=Path)
p.add_argument('--resume',type=Path);p.add_argument('--candidates',type=int,default=16)
p.add_argument('--generations',type=int,default=2);p.add_argument('--seed',type=int,default=721301)
p.add_argument('--training-indices',type=int,nargs='+',default=[0,64])
p.add_argument('--sigma-fraction',type=float,default=.08)
p.add_argument('--ik-seeds',type=Path)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg,FullRacketEnvV4
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH,PARAMETER_NAMES
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.physics.native import quat_rotate

stopping=False
def stop(*_):
    global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def atomic_torch(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(data,temp);temp.replace(path)


def main():
    if bool(args.warm_start)==bool(args.resume):raise ValueError('Choose exactly one warm start or optimizer resume')
    if args.candidates<4 or args.generations<1 or not 0<args.sigma_fraction<=.5:raise ValueError('Invalid search settings')
    if not args.training_indices or len(set(args.training_indices))!=len(args.training_indices):raise ValueError('Distinct TRAIN indices required')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=json.loads(protocol_path.read_text())
    suite=protocol['training_suite'];case_path=ROOT/suite['path']
    assert hashlib.sha256(case_path.read_bytes()).hexdigest()==suite['sha256']
    training_cases=json.loads(case_path.read_text())['cases']
    if min(args.training_indices)<0 or max(args.training_indices)>=len(training_cases):raise ValueError('TRAIN index out of range')
    source=args.resume or args.warm_start;prior=torch.load(source,map_location='cpu',weights_only=False)
    template=prior['controller_template'] if args.resume else dict(prior)
    if not args.resume:
        template['seed']=dict(template['seed'],smooth_recovery_arm=True)
        template['control_revision']='Continuous planned arm fold across leg-recovery transition; no target reset discontinuity'
    if args.ik_seeds:
        if args.resume:raise ValueError('Changing geometric seed requires warm-start and remeasured fitness')
        seed=json.loads(args.ik_seeds.read_text())['seeds'][0].copy()
        seed['pitch_compensation_basis_rad_per_rad']=FullRacketGeometryV4().pitch_compensation_basis(
            seed['joint_position_rad'],uv=seed.get('string_uv_m',[0.,0.]),face=seed.get('face_sign',1)).tolist()
        seed['smooth_recovery_arm']=True;template['seed']=seed
        template['parameters']=template['parameters'].clone();template['parameters'][8:14]=0.;template['parameters'][7]=1.
        if len(template['parameters'])>=25:template['parameters'][24]=seed['desired_ball_center_m'][0]
        template['event_calibration_s']=0.
        template['initial_timing_scope']='New geometric seed: zero timing calibration is an initialization assumption; residual learned on actual TRAIN rollouts'
    assert template['asset_sha256']==protocol['asset']['sha256']
    assert hashlib.sha256((ROOT/protocol['asset']['path']).read_bytes()).hexdigest()==protocol['asset']['sha256']
    if len(template['parameters'])!=24 and len(template['parameters'])!=26:raise ValueError('Use a24/26 parameter source')
    if len(template['parameters'])==24:
        template['parameters']=torch.cat((template['parameters'],torch.tensor([template['seed']['desired_ball_center_m'][0],0.])))
    template['event_calibration_s']=template.get('event_calibration_s',2.75-template['result']['event_time_s'])
    case_count=len(args.training_indices);count=args.candidates*case_count
    cases=[training_cases[i] for _ in range(args.candidates) for i in args.training_indices]
    cfg=full_racket_cfg(count,protocol['physics_dt_s']);cfg.seed=args.seed
    env=FullRacketEnvV4(cfg,cases=cases);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    controller=FullRacketRallyControllerV4(template,count,env.device)
    surface_offset=controller.teacher.contact_offset.clone()
    surface_offset[0]=controller.teacher.face_sign*env.racket_config['assumptions']['string_thickness_m']/2
    low=torch.tensor([*LOW,.80,-.12],device=env.device);high=torch.tensor([*HIGH,1.45,.12],device=env.device)
    mean=template['parameters'].to(env.device).clamp(low,high);reference=mean.clone()
    sigma=(high-low)*args.sigma_fraction
    rng=torch.Generator(device=env.device).manual_seed(args.seed)
    best_rank=(-1,-float('inf'));best_parameters=None;best_result=None;history=[];previous=0
    if args.resume:
        assert prior['schema']=='full-racket-ball-search-state-v4'
        assert prior['training_case_indices']==args.training_indices and prior['candidate_count']==args.candidates
        assert prior['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        assert prior['physics_dt_s']==cfg.sim.dt and prior['selection_contract']=='legal-first-ball-v4'
        mean=prior['mean'].to(env.device);sigma=prior['sigma'].to(env.device);rng.set_state(prior['rng'].cpu())
        previous=prior['generations'];best_result=prior['best_result'];reference=prior['reference_parameters'].to(env.device)
        if prior['best_parameters'] is not None:
            best_parameters=prior['best_parameters'].to(env.device);best_rank=tuple(prior['best_rank'])
    default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
    export_robot_geometry(get_current_stage(),env.robot,args.output)
    ticks=round(cfg.episode_length_s/.01)
    protocol_sha=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    sources=sorted(ROOT.glob('quadsmash/**/*.py'))+[Path(__file__).resolve()]
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if args.resume and prior['source_hashes']!=source_hashes:
        raise ValueError('Search/control source changed; use warm-start and remeasure fitness')
    base_record=dict(schema='full-racket-ball-search-state-v4',controller_template=template,
        protocol_sha256=protocol_sha,asset_sha256=template['asset_sha256'],physics_dt_s=cfg.sim.dt,
        candidate_count=args.candidates,training_case_indices=args.training_indices,training_suite=suite,
        reference_parameters=reference.cpu(),parameter_names=PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual'],
        selection_contract='legal-first-ball-v4',source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_hashes=source_hashes,formal_acceptance=False,formal_panels_read=False)
    try:
        for generation in range(args.generations):
            state=dict(base_record,mean=mean.cpu(),sigma=sigma.cpu(),rng=rng.get_state().cpu(),
                generations=previous+generation,best_parameters=best_parameters.cpu() if best_parameters is not None else None,
                best_result=best_result,best_rank=best_rank)
            atomic_torch(args.output/'generation_start.pt',state)
            population=(mean[None]+sigma[None]*torch.randn((args.candidates,len(low)),generator=rng,device=env.device)).clamp(low,high)
            population[0]=mean;population[1]=best_parameters if best_parameters is not None else reference
            env._episode_has_stepped.zero_();env._metrics_started=False
            env.rally_records.clear();env.episode_records.clear();env.contact_records.clear()
            observation,_=wrapper.reset();controller.reset();controller.parameters.copy_(population.repeat_interleave(case_count,dim=0))
            finished=torch.zeros(count,dtype=torch.bool,device=env.device)
            closest=torch.full((count,),10.,device=env.device);closest_time=torch.full((count,),float('nan'),device=env.device)
            closest_air=finished.clone();shaping=torch.full((count,),-100.,device=env.device)
            poses,public,actions,valid,trace=[],[],[],[],[]
            with torch.inference_mode():
                for tick in range(ticks):
                    obs=observation['policy'];data=env.robot.data
                    active=(~finished)&env.rally.in_play&env.launch_done
                    center=data.body_link_pos_w[:,env.racket_body_id];q=data.body_link_quat_w[:,env.racket_body_id]
                    gap=string_prism_gap(env.shuttle.data.root_pos_w,center,q,env.racket_config,cfg.shuttle_radius_m).clamp_min(0.)
                    improved=active&(gap<closest);closest[improved]=gap[improved];closest_time[improved]=obs[improved,775]
                    closest_air[improved]=env.observed_airborne[improved]
                    normal=controller.teacher.face_sign*quat_rotate(q,torch.tensor([1.,0,0],device=env.device).expand(count,-1))
                    surface_displacement=quat_rotate(q,surface_offset.expand(count,-1))
                    contact_velocity=data.body_link_lin_vel_w[:,env.racket_body_id]+torch.linalg.cross(data.body_link_ang_vel_w[:,env.racket_body_id],surface_displacement)
                    # Dense TRAIN reward is observed actual geometry/dynamics;
                    # native substep events remain authoritative for contacts.
                    score=-25*gap+2*env.observed_airborne.float()+.25*contact_velocity[:,0].clamp(-5,5)
                    score+=.15*(-contact_velocity[:,2]).clamp(-5,5)-.5*(normal[:,2]+.35).abs()
                    score-=10*env.body_ground_seen.float()
                    shaping=torch.where(active,torch.maximum(shaping,score),shaping)
                    action=controller.action(observation,default,limits)
                    public.append(obs[:case_count].cpu().numpy().copy());actions.append(action[:case_count].cpu().numpy().copy())
                    valid.append((~finished[:case_count]).cpu().numpy().copy())
                    if tick%2==0:
                        poses.append(torch.cat((data.body_link_pos_w[:case_count]-env.scene.env_origins[:case_count,None],data.body_link_quat_w[:case_count]),-1).cpu().numpy())
                        trace.append(dict(time_s=tick*.01,ball_position_m=(env.shuttle.data.root_pos_w[:case_count]-env.scene.env_origins[:case_count]).cpu().tolist(),
                            valid_first=(~finished[:case_count]).cpu().tolist(),in_play=env.rally.in_play[:case_count].cpu().tolist()))
                    observation,_,done,_=wrapper.step(action);finished|=done.bool()
                    if (tick+1)%100==0:
                        atomic_json(args.output/'progress.json',dict(generation=previous+generation,control_ticks=tick+1,
                            requested_ticks=ticks,first_episodes_complete=int(finished.sum()),formal_acceptance=False))
                        print(json.dumps(dict(generation=previous+generation,tick=tick+1,first_complete=int(finished.sum()))),flush=True)
                    if bool(finished.all()) or stopping:break
            np.savez_compressed(args.output/f'public_and_poses_{previous+generation:03d}.npz',observations=np.stack(public),
                actions=np.stack(actions),link_poses=np.stack(poses),valid_first_episode=np.stack(valid))
            atomic_json(args.output/f'trajectory_{previous+generation:03d}.json',trace)
            atomic_json(args.output/f'contacts_{previous+generation:03d}.json',env.contact_records)
            if stopping and not bool(finished.all()):
                atomic_json(args.output/'interrupted.json',dict(generation=previous+generation,control_ticks=tick+1,
                    resume_from='generation_start.pt',resume_semantics='Replay this generation from ground',formal_acceptance=False));return
            first,jumps={},{}
            for row in env.rally_records:first.setdefault(row['env_id'],row)
            for row in env.episode_records:jumps.setdefault(row['env_id'],row)
            assert len(first)==len(jumps)==count
            rows=[]
            for i in range(count):
                row=dict(first[i],jump=jumps[i],candidate_id=i//case_count,source_training_index=args.training_indices[i%case_count],
                    closest_string_gap_m=float(closest[i]),closest_gap_time_s=float(closest_time[i]),closest_gap_airborne=bool(closest_air[i]))
                fault=any(row[name] for name in env.rally.flags if name.endswith('_fault') or name=='engineering_body_crossing')
                legal_contact=row['legal_initial_contact'] and row['string_hit']
                outgoing=np.array(row['outgoing_velocity_mps'])
                row['attack_outgoing']=bool(row['released'] and legal_contact and row['airborne_string_hit']
                    and np.isfinite(outgoing).all() and np.linalg.norm(outgoing)>=8. and outgoing[0]>0 and outgoing[2]<=-1.)
                reward=float(shaping[i])+12*legal_contact+8*row['airborne_string_hit']+6*row['released']+10*row['attack_outgoing']
                reward+=12*row['crossed_net']+15*row['landed_target']+10*row['recovery_success']+100*row['complete_smash']-30*fault
                row.update(training_score=reward,legal_episode=not fault and row['numerical_failures']==0
                    and row['active_rally_launch_count']==1 and not row['jump']['terminated'])
                rows.append(row)
            groups=[rows[i*case_count:(i+1)*case_count] for i in range(args.candidates)]
            scores=torch.tensor([min(r['training_score'] for r in g)*.7+np.mean([r['training_score'] for r in g])*.3 for g in groups],device=env.device)
            feasible=torch.tensor([all(r['legal_episode'] for r in g) for g in groups],device=env.device)
            selection=torch.where(feasible,scores,scores-1000.) if bool(feasible.any()) else scores
            top=int(selection.argmax());rank=(int(feasible[top]),float(scores[top]))
            if rank>best_rank:best_rank=rank;best_parameters=population[top].clone();best_result=groups[top]
            pool=int(feasible.sum());elite=selection.topk(min(pool,max(2,args.candidates//8)) if pool else max(2,args.candidates//8)).indices
            mean=.3*mean+.7*population[elite].mean(0)
            sigma=(.3*sigma+.7*population[elite].std(0,unbiased=False)).clamp_min((high-low)*.025)
            entry=dict(generation=previous+generation,episodes=count,control_ticks=tick+1,env_steps=(tick+1)*count,
                legal_candidates=int(feasible.sum()),best_candidate=top,
                best_score=float(scores[top]),contacts=sum(r['had_contact'] for r in rows),airborne_string_hits=sum(r['airborne_string_hit'] for r in rows),
                complete=sum(r['complete_smash'] for r in rows),minimum_gap_m=min(r['closest_string_gap_m'] for r in rows),formal_acceptance=False)
            history.append(entry);atomic_json(args.output/f'episodes_{previous+generation:03d}.json',rows)
            state=dict(base_record,parameters=population.cpu(),rows=rows,mean=mean.cpu(),sigma=sigma.cpu(),rng=rng.get_state().cpu(),
                generations=previous+generation+1,best_parameters=best_parameters.cpu(),best_result=best_result,best_rank=best_rank)
            atomic_torch(args.output/f'population_{previous+generation:03d}.pt',state);atomic_torch(args.output/'resume.pt',state)
            learned=dict(template,schema='full-racket-ball-cem-v4',parameters=best_parameters.cpu(),training_result=best_result,
                training_suite=suite,training_case_indices=args.training_indices,parameter_names=base_record['parameter_names'],
                event_calibration_s=template['event_calibration_s'],formal_acceptance=False,kind='Actual ball TRAIN optimization; no formal pass')
            atomic_torch(args.output/'best_candidate.pt',learned)
            atomic_json(args.output/'metrics.json',dict(history=history,best_training_episodes=best_result,
                initialization_no_ball_result=template['result'],event_calibration_s=template['event_calibration_s'],
                initial_timing_scope=template.get('initial_timing_scope','Previously measured no-ball timing'),
                formal_acceptance=False,formal_panels_read=False))
            print(json.dumps(entry),flush=True)
            if stopping:break
        if len(history)==args.generations:
            atomic_json(args.output/'completed.json',dict(env_steps=sum(x['env_steps'] for x in history)))
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
