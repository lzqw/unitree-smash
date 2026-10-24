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
p.add_argument('--walk-speed',type=float);p.add_argument('--waypoint-x',type=float)
p.add_argument('--timing-grid',action='store_true',help='Reserve12 candidates for measured leg timing/arm speed/Cartesian-gain combinations')
p.add_argument('--push-grid',action='store_true',help='Reserve16 candidates for front/back thigh/calf push coordination')
p.add_argument('--contact-curriculum',action='store_true',help='TRAIN: retain actual legal airborne strings contacts while optimizing failed return and recovery; formal flags unchanged')
p.add_argument('--retain-contact-recovery',action='store_true',help='TRAIN: retain an actual same-episode legal airborne strings contact and qualified recovery ahead of additional contacts ending in falls; native task labels unchanged')
p.add_argument('--contact-grid',action='store_true',help='Reserve12 candidates for arm lead/speed around a measured contact, preserving leg timing')
p.add_argument('--pitch-grid',action='store_true',help='Reserve12 candidates spanning existing arm pitch-compensation gain and timing; original joint/actuator limits remain')
p.add_argument('--search-arm-velocity',action='store_true',help='Append six bounded arm velocity seed coordinates to the26 controller parameters')
p.add_argument('--arm-velocity-bank',type=Path,help='Four offline velocity proposals for a12-candidate velocity/timing grid; requires --search-arm-velocity')
p.add_argument('--interception-plane-grid',action='store_true',help='Expand velocity/timing bank to36 candidates across public interception planes1.10/1.25/1.40m; requires --arm-velocity-bank')
p.add_argument('--candidate-bank',type=Path,help='Validated bounded whole-body proposals, generation0only; preserve two references and let later generations adapt freely')
p.add_argument('--prepose-gain',type=float,help='Optional current-public-forecast preparation correction in[0,1]; warm start only')
p.add_argument('--recovery-leg-bank',type=Path,help='Optional measured same-family recovery26-vector: reserve8 contact-arm/recovery-leg crossover candidates')
p.add_argument('--continue-after-miss',action='store_true',help='TRAIN curriculum: observe landing after missed/blocked/backward shuttle return; all rally failures remain failures, robot faults terminate')
p.add_argument('--push-search-range',type=float,default=.15,help='Symmetric front/back leg target-offset search range in rad; native actuator/joint limits stay unchanged')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg,FullRacketEnvV4
from quadsmash.tasks.stand.full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH,PARAMETER_NAMES
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_arm_search_v4 import (
    VELOCITY_PARAMETER_NAMES,extend_arm_velocity_parameters,arm_velocity_grid,interception_velocity_grid,body_crossovers_preserving_velocity)
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank,install_candidate_bank
from quadsmash.control.full_racket_contact_curriculum_v4 import (
    contact_search_tiers,actual_outgoing_shaping,arm_timing_velocity_grid,measured_recovery_leg_crossovers,pitch_compensation_grid)
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.physics.native import quat_rotate
from quadsmash.usd_dependencies_v4 import audit_composed_asset

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
    if args.timing_grid and args.candidates<14:raise ValueError('Timing grid requires at least14 candidates')
    if args.push_grid and args.candidates<18:raise ValueError('Push grid requires at least18 candidates')
    if sum([args.timing_grid,args.push_grid,args.contact_grid,args.pitch_grid,bool(args.arm_velocity_bank),bool(args.candidate_bank)])>1:raise ValueError('Choose one controlled grid')
    if args.candidate_bank and (not args.contact_curriculum or args.recovery_leg_bank):
        raise ValueError('Candidate bank requires contact curriculum and supplies its own body proposals')
    if args.arm_velocity_bank and (not args.search_arm_velocity or not args.contact_curriculum or args.candidates<14):
        raise ValueError('Velocity bank requires32-parameter contact curriculum and at least14 candidates')
    if args.interception_plane_grid and (not args.arm_velocity_bank or args.candidates < 38 + 8*bool(args.recovery_leg_bank)):
        raise ValueError('Interception grid requires velocity bank and38 candidates, or46 with recovery crossover')
    if args.prepose_gain is not None and (args.resume or not 0.<=args.prepose_gain<=1.):
        raise ValueError('Preparation gain in[0,1] requires a fresh warm start')
    if (args.contact_grid or args.pitch_grid) and (not args.contact_curriculum or args.candidates<14):
        raise ValueError('Contact grid requires contact curriculum and at least14 candidates')
    if args.contact_curriculum and not args.continue_after_miss:
        raise ValueError('Contact curriculum requires observing recovery after failed shuttle returns')
    if args.retain_contact_recovery and not args.contact_curriculum:
        raise ValueError('Retaining contact/recovery requires contact curriculum')
    if args.recovery_leg_bank and (not (args.contact_grid or args.pitch_grid or args.arm_velocity_bank) or args.candidates<22):
        raise ValueError('Recovery leg crossover requires contact grid and at least22 candidates')
    if not .15<=args.push_search_range<=.5:raise ValueError('Push search range must be between .15 and .5rad')
    sampling_contract=('bounded-candidate-bank-generation0-v1' if args.candidate_bank else 'contact-pitch-compensation-grid-v1' if args.pitch_grid else 'contact-arm-timing-speed-grid-v1' if args.contact_grid else 'front-back-push-grid-v1' if args.push_grid else
        'timing-velocity-cartesian-grid-v1' if args.timing_grid else 'gaussian-cem-v1')
    selection_contract=('actual-contact-recovery-chain-v2' if args.retain_contact_recovery else
        'actual-contact-then-return-and-recovery-v1' if args.contact_curriculum else
        'ball-with-jump-recovery-v3' if args.continue_after_miss else 'legal-first-ball-v4')
    if args.recovery_leg_bank:sampling_contract+='-measured-recovery-leg-crossover-v1'
    if args.search_arm_velocity:sampling_contract+='-joint-velocity32-v1'
    if args.interception_plane_grid:sampling_contract+='-interception-plane-grid36-v1'
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
    if args.prepose_gain is not None:
        template['seed']=dict(template['seed'],causal_arm_prepose_gain=args.prepose_gain)
    if args.ik_seeds:
        if args.resume:raise ValueError('Changing geometric seed requires warm-start and remeasured fitness')
        seed=json.loads(args.ik_seeds.read_text())['seeds'][0].copy()
        seed['pitch_compensation_basis_rad_per_rad']=FullRacketGeometryV4().pitch_compensation_basis(
            seed['joint_position_rad'],uv=seed.get('string_uv_m',[0.,0.]),face=seed.get('face_sign',1)).tolist()
        seed['smooth_recovery_arm']=True;template['seed']=seed
        template['parameters']=template['parameters'].clone();template['parameters'][8:14]=0.;template['parameters'][7]=1.
        if len(template['parameters'])>=25:template['parameters'][24]=seed['desired_ball_center_m'][0]
        if len(template['parameters'])==32:template['parameters'][26:32]=torch.tensor(seed['joint_velocity_seed_rad_s'])
        template['event_calibration_s']=0.
        template['initial_timing_scope']='New geometric seed: zero timing calibration is an initialization assumption; residual learned on actual TRAIN rollouts'
    assert template['asset_sha256']==protocol['asset']['sha256']
    assert hashlib.sha256((ROOT/protocol['asset']['path']).read_bytes()).hexdigest()==protocol['asset']['sha256']
    if len(template['parameters']) not in (24,26,32):raise ValueError('Use a24/26/32 parameter source')
    if len(template['parameters'])==32 and not args.search_arm_velocity:
        raise ValueError('A32-parameter source requires --search-arm-velocity')
    if len(template['parameters'])==24:
        template['parameters']=torch.cat((template['parameters'],torch.tensor([template['seed']['desired_ball_center_m'][0],0.])))
    if args.search_arm_velocity:
        template['parameters']=extend_arm_velocity_parameters(template['parameters'],template['seed']['joint_velocity_seed_rad_s'])
    velocity_bank=None
    if args.arm_velocity_bank:
        velocity_bank=json.loads(args.arm_velocity_bank.read_text())
        if velocity_bank['asset_sha256']!=template['asset_sha256']:raise ValueError('Velocity bank asset mismatch')
        if velocity_bank['reference_seed']!=template['seed']:raise ValueError('Velocity bank controller family mismatch')
    leg_bank=None
    if args.recovery_leg_bank:
        leg_bank=json.loads(args.recovery_leg_bank.read_text())
        if leg_bank['seed']!=template['seed']:raise ValueError('Recovery crossover requires the same original arm/controller family')
        if leg_bank['asset_sha256']!=template['asset_sha256']:raise ValueError('Recovery bank asset mismatch')
    template['event_calibration_s']=template.get('event_calibration_s',2.75-template['result']['event_time_s'])
    if args.walk_speed is not None or args.waypoint_x is not None:
        if args.resume:raise ValueError('Parameter overrides require warm-start and remeasured fitness')
        template['parameters']=template['parameters'].clone()
        if args.walk_speed is not None:template['parameters'][0]=args.walk_speed
        if args.waypoint_x is not None:template['parameters'][1]=args.waypoint_x
    case_count=len(args.training_indices);count=args.candidates*case_count
    cases=[training_cases[i] for _ in range(args.candidates) for i in args.training_indices]
    cfg=full_racket_cfg(count,protocol['physics_dt_s']);cfg.seed=args.seed
    env=RallyTrainingEnvV4(cfg,cases=cases,continue_after_miss=args.continue_after_miss);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    if env.robot.is_fixed_base:raise ValueError('Training requires a floating robot and ground initialization')
    atomic_json(args.output/'asset_dependencies.json',audit_composed_asset(ROOT,cfg.robot.spawn.usd_path))
    controller=FullRacketRallyControllerV4(template,count,env.device)
    surface_offset=controller.teacher.contact_offset.clone()
    surface_offset[0]=controller.teacher.face_sign*env.racket_config['assumptions']['string_thickness_m']/2
    low=torch.tensor([*LOW,.80,-.12],device=env.device);high=torch.tensor([*HIGH,1.45,.12],device=env.device)
    if args.search_arm_velocity:
        low=torch.cat((low,torch.full((6,),-5.,device=env.device)))
        high=torch.cat((high,torch.full((6,),5.,device=env.device)))
    low[22:24]=-args.push_search_range;high[22:24]=args.push_search_range
    mean=template['parameters'].to(env.device).clamp(low,high);reference=mean.clone()
    sigma=(high-low)*args.sigma_fraction
    rng=torch.Generator(device=env.device).manual_seed(args.seed)
    best_rank=(-1,-float('inf'));best_parameters=None;best_result=None;history=[];previous=0
    if args.resume:
        assert prior['schema']=='full-racket-ball-search-state-v4'
        assert prior['training_case_indices']==args.training_indices and prior['candidate_count']==args.candidates
        assert prior['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        assert prior['physics_dt_s']==cfg.sim.dt and prior['selection_contract']==selection_contract
        assert prior.get('push_search_range',.15)==args.push_search_range,'Search bounds changed; use warm-start'
        assert prior.get('sampling_contract','gaussian-cem-v1')==sampling_contract,'Sampling changed; use warm-start'
        mean=prior['mean'].to(env.device);sigma=prior['sigma'].to(env.device);rng.set_state(prior['rng'].cpu())
        previous=prior['generations'];best_result=prior['best_result'];reference=prior['reference_parameters'].to(env.device)
        if prior['best_parameters'] is not None:
            best_parameters=prior['best_parameters'].to(env.device);best_rank=tuple(prior['best_rank'])
    candidate_bank=None
    if args.candidate_bank:
        candidate_bank=validate_candidate_bank(json.loads(args.candidate_bank.read_text()),template['seed'],template['asset_sha256'],
            PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual']+(VELOCITY_PARAMETER_NAMES if args.search_arm_velocity else []),
            reference.cpu().numpy(),low.cpu().numpy(),high.cpu().numpy(),args.candidates-2)
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
        reference_parameters=reference.cpu(),parameter_names=PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual']+(VELOCITY_PARAMETER_NAMES if args.search_arm_velocity else []),
        selection_contract=selection_contract,source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sampling_contract=sampling_contract,
        push_search_range=args.push_search_range,
        source_hashes=source_hashes,formal_acceptance=False,formal_panels_read=False,
        arm_velocity_search_enabled=args.search_arm_velocity,
        trajectory_contract='All lanes native50Hz poses/ball/clock/first-episode validity; exact terminal/contact snapshots separate',
        training_continue_after_miss=args.continue_after_miss,
        training_retain_contact_recovery=args.retain_contact_recovery,
        training_early_termination=('Robot-rule fault, body ground or fallen; observe recovery after failed shuttle return without reviving its rally' if args.continue_after_miss else
            'Rally fault, first ground without target, body ground or fallen')+'; formal evaluator unchanged')
    if leg_bank is not None:
        bank_sha=hashlib.sha256(args.recovery_leg_bank.read_bytes()).hexdigest()
        if args.resume and prior.get('recovery_leg_bank_sha256')!=bank_sha:raise ValueError('Recovery bank changed; use warm-start')
        base_record.update(recovery_leg_bank_path=str(args.recovery_leg_bank),recovery_leg_bank_sha256=bank_sha)
    if velocity_bank is not None:
        bank_sha=hashlib.sha256(args.arm_velocity_bank.read_bytes()).hexdigest()
        if args.resume and prior.get('arm_velocity_bank_sha256')!=bank_sha:raise ValueError('Velocity bank changed; use warm-start')
        base_record.update(arm_velocity_bank_path=str(args.arm_velocity_bank),arm_velocity_bank_sha256=bank_sha)
    if candidate_bank is not None:
        bank_sha=hashlib.sha256(args.candidate_bank.read_bytes()).hexdigest()
        if args.resume and prior.get('candidate_bank_sha256')!=bank_sha:raise ValueError('Candidate bank changed; use warm-start')
        base_record.update(candidate_bank_path=str(args.candidate_bank),candidate_bank_sha256=bank_sha,
                          candidate_bank_rows=len(candidate_bank),candidate_bank_schedule='generation0only')
    try:
        for generation in range(args.generations):
            state=dict(base_record,mean=mean.cpu(),sigma=sigma.cpu(),rng=rng.get_state().cpu(),
                generations=previous+generation,best_parameters=best_parameters.cpu() if best_parameters is not None else None,
                best_result=best_result,best_rank=best_rank)
            atomic_torch(args.output/'generation_start.pt',state)
            atomic_torch(args.output/f'generation_start_{previous+generation:03d}.pt',state)
            population=(mean[None]+sigma[None]*torch.randn((args.candidates,len(low)),generator=rng,device=env.device)).clamp(low,high)
            population[0]=mean;population[1]=best_parameters if best_parameters is not None else reference
            if args.timing_grid:
                # Actual motor/body coordination, on the same frozen TRAIN
                # balls. Reference candidates remain unchanged and every
                # controller is evaluated by the same native event chain.
                index=2
                for phase in [1.28,1.38,1.48]:
                    for speed in [.55,.85]:
                        for servo_gain in [.4,.8]:
                            population[index]=mean
                            population[index,2]=phase;population[index,7]=speed;population[index,18]=servo_gain
                            index+=1
            if args.push_grid:
                index=2
                for thigh in [-.15,-.05,.05,.15]:
                    for calf in [-.15,-.05,.05,.15]:
                        population[index]=mean
                        population[index,22]=thigh;population[index,23]=calf
                        index+=1
            if args.contact_grid:
                grid=arm_timing_velocity_grid(reference.cpu().numpy(),low.cpu().numpy(),high.cpu().numpy())
                population[2:14]=torch.as_tensor(grid,device=env.device,dtype=population.dtype)
            if args.pitch_grid:
                grid=pitch_compensation_grid(reference.cpu().numpy(),low.cpu().numpy(),high.cpu().numpy())
                population[2:14]=torch.as_tensor(grid,device=env.device,dtype=population.dtype)
            body_start=14
            if velocity_bank is not None:
                builder=interception_velocity_grid if args.interception_plane_grid else arm_velocity_grid
                grid=builder(reference.cpu().numpy(),velocity_bank['joint_velocity_seeds_rad_s'],low.cpu().numpy(),high.cpu().numpy())
                population[2:2+len(grid)]=torch.as_tensor(grid,device=env.device,dtype=population.dtype)
                body_start=2+len(grid)
            if leg_bank is not None:
                grid=body_crossovers_preserving_velocity(reference.cpu().numpy(),leg_bank['parameters'],low.cpu().numpy(),high.cpu().numpy())
                population[body_start:body_start+len(grid)]=torch.as_tensor(grid,device=env.device,dtype=population.dtype)
            if candidate_bank is not None:
                install_candidate_bank(population,candidate_bank,previous+generation)
            env._episode_has_stepped.zero_();env._metrics_started=False
            env.rally_records.clear();env.episode_records.clear();env.contact_records.clear()
            env.racket_plane_records.clear();env.episode_index.fill_(-1)
            observation,_=wrapper.reset();controller.reset();controller.parameters.copy_(population.repeat_interleave(case_count,dim=0))
            assert bool((env.episode_index==0).all()),'Each generation must start fresh local episode identifiers'
            finished=torch.zeros(count,dtype=torch.bool,device=env.device)
            closest=torch.full((count,),10.,device=env.device);closest_time=torch.full((count,),float('nan'),device=env.device)
            closest_air=finished.clone();shaping=torch.full((count,),-100.,device=env.device)
            poses,public,actions,valid,trace=[],[],[],[],[]
            all_poses,all_balls,all_clocks,all_valid=[],[],[],[]
            window_poses,window_balls,window_clocks,window_valid,window_observations=[],[],[],[],[]
            control_clock,control_actions,control_previous,control_valid,control_recovery=[],[],[],[],[]
            def save_control_window(name):
                if control_clock:
                    np.savez_compressed(args.output/name,clock_s=np.stack(control_clock),
                        actions=np.stack(control_actions),observed_previous_actions=np.stack(control_previous),
                        valid_first_episode=np.stack(control_valid),recovery_active=np.stack(control_recovery))
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
                    if 2.2<=tick*.01<=3.5:
                        local=(obs[:,775]-controller.parameters[:,2]-controller.teacher.leg_event_time_gain*(controller.event_time-2.75))*controller.parameters[:,4]
                        control_clock.append(obs[:,775].cpu().numpy().copy())
                        control_actions.append(action.cpu().numpy().copy())
                        control_previous.append(obs[:,-18:].cpu().numpy().copy())
                        control_valid.append((~finished).cpu().numpy().copy())
                        control_recovery.append((local>=controller.parameters[:,3]).cpu().numpy().copy())
                    public.append(obs[:case_count].cpu().numpy().copy());actions.append(action[:case_count].cpu().numpy().copy())
                    valid.append((~finished[:case_count]).cpu().numpy().copy())
                    if tick%2==0:
                        all_poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],data.body_link_quat_w),-1).cpu().numpy())
                        all_balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                        all_clocks.append(obs[:,775].cpu().numpy().copy())
                        all_valid.append((~finished).cpu().numpy().copy())
                        poses.append(torch.cat((data.body_link_pos_w[:case_count]-env.scene.env_origins[:case_count,None],data.body_link_quat_w[:case_count]),-1).cpu().numpy())
                        trace.append(dict(time_s=tick*.01,ball_position_m=(env.shuttle.data.root_pos_w[:case_count]-env.scene.env_origins[:case_count]).cpu().tolist(),
                            valid_first=(~finished[:case_count]).cpu().tolist(),in_play=env.rally.in_play[:case_count].cpu().tolist()))
                        if 2.2<=tick*.01<=3.4:
                            window_poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],data.body_link_quat_w),-1).cpu().numpy())
                            window_balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                            window_clocks.append(obs[:,775].cpu().numpy().copy())
                            window_valid.append((~finished).cpu().numpy().copy())
                            window_observations.append(obs.cpu().numpy().copy())
                    observation,_,done,_=wrapper.step(action);finished|=done.bool()
                    if (tick+1)%100==0:
                        first_partial={}
                        for row in env.rally_records:first_partial.setdefault(row['env_id'],row)
                        atomic_json(args.output/'progress.json',dict(generation=previous+generation,control_ticks=tick+1,
                            requested_ticks=ticks,first_episodes_complete=int(finished.sum()),
                            first_contacts=sum(r['had_contact'] for r in first_partial.values())+int(((~finished)&env.rally.had_contact).sum()),
                            active_first_rallies=int(((~finished)&env.rally.in_play).sum()),
                            closest_string_gaps_m=closest.cpu().tolist(),partial=True,formal_acceptance=False))
                        atomic_json(args.output/'episodes.partial.json',env.rally_records)
                        atomic_json(args.output/'contacts.partial.json',env.contact_records)
                        atomic_json(args.output/'causal_plans.partial.json',controller.logs)
                        atomic_torch(args.output/'partial_search_state.pt',dict(parameters=population.cpu(),
                            closest_string_gap_m=closest.cpu(),closest_gap_time_s=closest_time.cpu(),
                            closest_gap_airborne=closest_air.cpu(),first_finished=finished.cpu(),
                            generation=previous+generation,control_ticks=tick+1,formal_acceptance=False))
                        np.savez_compressed(args.output/'partial_public_and_poses.npz',observations=np.stack(public),
                            actions=np.stack(actions),link_poses=np.stack(poses),valid_first_episode=np.stack(valid))
                        atomic_json(args.output/'trajectory.partial.json',trace)
                        np.savez_compressed(args.output/'all_first_episodes.partial.npz',link_poses=np.stack(all_poses),
                            ball_position_m=np.stack(all_balls),episode_clock_s=np.stack(all_clocks),
                            valid_first_episode=np.stack(all_valid))
                        if window_poses:
                            np.savez_compressed(args.output/'near_contact_all.partial.npz',link_poses=np.stack(window_poses),
                                ball_position_m=np.stack(window_balls),episode_clock_s=np.stack(window_clocks),
                                valid_first_episode=np.stack(window_valid),observations=np.stack(window_observations))
                        save_control_window('control_window.partial.npz')
                        print(json.dumps(dict(generation=previous+generation,tick=tick+1,first_complete=int(finished.sum()))),flush=True)
                    if bool(finished.all()) or stopping:break
            np.savez_compressed(args.output/f'public_and_poses_{previous+generation:03d}.npz',observations=np.stack(public),
                actions=np.stack(actions),link_poses=np.stack(poses),valid_first_episode=np.stack(valid))
            atomic_json(args.output/f'trajectory_{previous+generation:03d}.json',trace)
            atomic_json(args.output/f'contacts_{previous+generation:03d}.json',env.contact_records)
            atomic_json(args.output/f'causal_plans_{previous+generation:03d}.json',controller.logs)
            np.savez_compressed(args.output/f'all_first_episodes_{previous+generation:03d}.npz',link_poses=np.stack(all_poses),
                ball_position_m=np.stack(all_balls),episode_clock_s=np.stack(all_clocks),valid_first_episode=np.stack(all_valid))
            if window_poses:
                np.savez_compressed(args.output/f'near_contact_all_{previous+generation:03d}.npz',link_poses=np.stack(window_poses),
                    ball_position_m=np.stack(window_balls),episode_clock_s=np.stack(window_clocks),
                    valid_first_episode=np.stack(window_valid),observations=np.stack(window_observations))
            save_control_window(f'control_window_{previous+generation:03d}.npz')
            if stopping and not bool(finished.all()):
                # Keep the latest terminal events and actual population even
                # when stopping between the100tick periodic snapshots.
                suffix=f'{previous+generation:03d}'
                atomic_json(args.output/f'interrupted_rallies_{suffix}.json',env.rally_records)
                atomic_json(args.output/f'interrupted_jumps_{suffix}.json',env.episode_records)
                atomic_torch(args.output/f'interrupted_search_state_{suffix}.pt',dict(
                    parameters=population.cpu(),first_finished=finished.cpu(),
                    closest_string_gap_m=closest.cpu(),closest_gap_time_s=closest_time.cpu(),
                    closest_gap_airborne=closest_air.cpu(),generation=previous+generation,
                    control_ticks=tick+1,formal_acceptance=False))
                atomic_json(args.output/'interrupted.json',dict(generation=previous+generation,control_ticks=tick+1,
                    resume_from='generation_start.pt',resume_semantics='Replay this generation from ground',
                    terminal_rally_records=f'interrupted_rallies_{suffix}.json',
                    terminal_jump_records=f'interrupted_jumps_{suffix}.json',
                    actual_partial_population=f'interrupted_search_state_{suffix}.pt',formal_acceptance=False));return
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
                if args.continue_after_miss:
                    # This curriculum supplies landing feedback even after a
                    # miss. Actual complete/attack/legal/landing flags remain
                    # untouched; no dead-ball return can become successful.
                    jump=row['jump']
                    recovery_shaping=(6*min(jump['recovery_s']/.8,1.)+8*jump['qualified_jump_landed']
                        -15*jump['body_landing']-5*row['training_terminal_reasons']['root_fallen'])
                    reward+=recovery_shaping
                    row['training_recovery_shaping']=recovery_shaping
                if args.contact_curriculum:
                    row['actual_outgoing_training_shaping']=actual_outgoing_shaping(row)
                    reward+=row['actual_outgoing_training_shaping']
                row.update(training_score=reward,legal_episode=not fault and row['numerical_failures']==0
                    and row['active_rally_launch_count']==1 and not row['jump']['terminated'])
                rows.append(row)
            groups=[rows[i*case_count:(i+1)*case_count] for i in range(args.candidates)]
            scores=torch.tensor([min(r['training_score'] for r in g)*.7+np.mean([r['training_score'] for r in g])*.3 for g in groups],device=env.device)
            feasible=torch.tensor([all(r['legal_episode'] for r in g) for g in groups],device=env.device)
            if args.contact_curriculum:
                tiers=torch.as_tensor(contact_search_tiers(groups,retain_recovery=args.retain_contact_recovery),device=env.device)
                if int(tiers.max())<0:raise RuntimeError('No numerically valid contact-curriculum candidate')
                eligible=tiers==tiers.max()
                selection=torch.where(eligible,scores,torch.full_like(scores,-float('inf')))
            else:
                selection=torch.where(feasible,scores,scores-1000.) if bool(feasible.any()) else scores
            top=int(selection.argmax());rank=(int(tiers[top]) if args.contact_curriculum else int(feasible[top]),float(scores[top]))
            if rank>best_rank:best_rank=rank;best_parameters=population[top].clone();best_result=groups[top]
            pool=int(eligible.sum()) if args.contact_curriculum else int(feasible.sum())
            elite=selection.topk(min(pool,max(2,args.candidates//8)) if pool else max(2,args.candidates//8)).indices
            mean=.3*mean+.7*population[elite].mean(0)
            sigma=(.3*sigma+.7*population[elite].std(0,unbiased=False)).clamp_min((high-low)*.025)
            entry=dict(generation=previous+generation,episodes=count,control_ticks=tick+1,env_steps=(tick+1)*count,
                legal_candidates=int(feasible.sum()),best_candidate=top,
                best_score=float(scores[top]),contacts=sum(r['had_contact'] for r in rows),airborne_string_hits=sum(r['airborne_string_hit'] for r in rows),
                complete=sum(r['complete_smash'] for r in rows),minimum_gap_m=min(r['closest_string_gap_m'] for r in rows),formal_acceptance=False)
            if args.contact_curriculum:
                entry.update(contact_search_tiers=tiers.cpu().tolist(),best_contact_search_tier=int(tiers[top]),
                    selection_contract=selection_contract)
            history.append(entry);atomic_json(args.output/f'episodes_{previous+generation:03d}.json',rows)
            state=dict(base_record,parameters=population.cpu(),rows=rows,mean=mean.cpu(),sigma=sigma.cpu(),rng=rng.get_state().cpu(),
                generations=previous+generation+1,best_parameters=best_parameters.cpu(),best_result=best_result,best_rank=best_rank)
            atomic_torch(args.output/f'population_{previous+generation:03d}.pt',state);atomic_torch(args.output/'resume.pt',state)
            learned=dict(template,schema='full-racket-ball-cem-v4',parameters=best_parameters.cpu(),training_result=best_result,
                training_suite=suite,training_case_indices=args.training_indices,parameter_names=base_record['parameter_names'],
                event_calibration_s=template['event_calibration_s'],native_execution=True,performance_inherited=False,
                formal_acceptance=False,kind='Actual ball TRAIN optimization; no formal pass')
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
