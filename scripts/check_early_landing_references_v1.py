"""Test interpretable recovery references before the recorded head-proximity event.

Queries are actual ground-start observations, never simulated resets. These are
offline warm-start diagnostics, not a deployed trigger or native teacher. The
recorded student continuation is an offline comparator, not causal foresight.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.compare_frozen_complete5_v2 import read_run
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.loaded_pinocchio_model_v1 import rotation_wxyz
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.feedback_tail_rollout_v1 import terminal_measures,terminal_violations,kinematically_stable
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
from scripts.audit_joint_sequence_proposal_v1 import plain


def rollout(controller,initial,feedback,nominal,mode,spheres,geometry,protocol,*,tick,substeps):
    state=copy.deepcopy(initial);length=800-tick;started=time.monotonic()
    # The interface default is provided explicitly with the recorded inputs.
    default=np.asarray(nominal['default'])
    constant=initial.previous_command_target.copy()
    standing=constant.copy();standing[:12]=default[:12]
    measured_arm_standing=standing.copy();measured_arm_standing[12:]=initial.joint_position[12:]
    targets=[];states=[];diagnostics={};failure=None;trailing=0;longest=0;end=None
    for step in range(length):
        if time.monotonic()-started>300.:
            failure=dict(step=step,reason='bounded_model_runtime');break
        previous=state.previous_command_target
        if mode=='recorded_student':target=nominal['commands'][tick+step].copy()
        elif mode=='hold_actual_previous':target=constant.copy()
        elif mode=='recorded_stand_legs_hold_arm':target=standing.copy()
        elif mode=='stand_legs_measured_arm':target=measured_arm_standing.copy()
        elif mode=='terminal_legs_measured_arm':
            target,_=feedback.action(min(80+step,339),state,previous)
            target[12:]=initial.joint_position[12:]
        elif mode=='existing_terminal_feedback':target,_=feedback.action(min(80+step,339),state,previous)
        else:raise ValueError('Unknown reference')
        target=np.clip(target,controller.qp.interface.limits[:,0],controller.qp.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*controller.qp.interface.arm_rate,
                           previous[12:]+.01*controller.qp.interface.arm_rate)
        try:
            result=predict_frictional_spheres(controller.qp.model,controller.qp.interface,state,target,
                spheres,substeps=substeps,geometry=geometry)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=step,reason='model_solver_rejected',error=repr(error));break
        d=result['diagnostics']
        for key,value in d.items():diagnostics[key]=max(diagnostics.get(key,0.),value)
        bad={k:v for k,v in d.items() if (k=='nonfoot_ground_violation_m' and v>1e-6) or
            (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            following=result['state'];interface=controller.qp.interface
            position_excess=np.maximum(interface.limits[:,0]-following.joint_position,
                                       following.joint_position-interface.limits[:,1])
            velocity_excess=abs(following.joint_velocity)-interface.velocity
            failure=dict(step=step,reason='path_constraints',violations=bad,
                first_model_ground_violation=result['first_ground_violation'],
                interval_end_joint_limit_violations=[dict(joint=controller.qp.model.joint_order[j],
                    position_rad=float(following.joint_position[j]),velocity_rad_s=float(following.joint_velocity[j]),
                    position_excess_rad=float(max(0.,position_excess[j])),
                    velocity_excess_rad_s=float(max(0.,velocity_excess[j]))) for j in range(18)
                    if position_excess[j]>3e-4 or velocity_excess[j]>3e-4],
                requested_target_rad=target.copy(),
                interval_end_q=following.q.copy(),interval_end_v=following.v.copy());break
        state=result['state'];state.previous_command_target=target.copy()
        end=terminal_measures(controller,state)
        stable=kinematically_stable(end,protocol) and state.q[2]>=.12
        trailing=trailing+1 if stable else 0;longest=max(longest,trailing)
        targets.append(target);states.append(dict(q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy()))
    violations=terminal_violations(end,.01) if end is not None else {'no_endpoint':True}
    if trailing<80:violations['trailing_model_window_s']=trailing*.01
    if state.q[2]<.12:violations['terminal_root_height_m']=float(state.q[2])
    accepted=failure is None and len(states)==length and not violations
    return dict(accepted=accepted,reference=mode,start_tick=tick,required_model_steps=length,
        completed_model_steps=len(states),failure=failure,terminal=end,terminal_violations=violations,
        longest_model_kinematic_window_s=longest*.01,trailing_model_kinematic_window_s=trailing*.01,
        diagnostics=diagnostics,wall_seconds=time.monotonic()-started,substeps=substeps,
        commands=targets if accepted else None,states=states if accepted else None,
        native_execution=False,valid_expert_labels=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','validation','warm-plan','sphere-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tick',type=int,default=300)
    parser.add_argument('--screen-substeps',type=int,choices=[16,160],default=16,
                        help='160 checks whether a coarse rejection persists at the native time step.')
    args=parser.parse_args()
    if args.output.exists() or not 280<=args.tick<=310 or args.tick%2:
        raise ValueError('Fresh, bounded, pose-aligned early query required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=read(args.bundle/'manifest.json');run=Path(manifest['baseline']);archive=args.bundle/manifest['bundle_name']
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    task=read(protocol_path);task['_root']=str(ROOT);summary,_=read_run(run,task)
    if summary['complete5']!=27:raise ValueError('Unexpected baseline')
    trace_manifest_path=run/'fixed_policy_traces/manifest.json';trace_manifest=read(trace_manifest_path)
    entry=next(e for e in trace_manifest['episodes'] if e['source_training_index']==3)
    trace_path=run/'fixed_policy_traces'/entry['data_file']
    if entry['env_id']!=3 or sha(trace_path)!=entry['data_sha256'] or sha(archive)!=manifest['bundle_sha256']:
        raise ValueError('Changed trace or bundle')
    with np.load(trace_path,allow_pickle=False) as data:
        public=unpack_public(data['packed_public']);actions=data['dispatched_actions'].copy()
    if len(public)!=800:raise ValueError('Complete recorded comparator required')
    with np.load(archive,allow_pickle=False) as data:old_query=data['case_003_query_public'].copy()
    controller=build_controller(manifest['configuration']);model=controller.qp.model
    feedback,_=load_verified_policy(model,controller,old_query,args.warm_plan,args.validation)
    native_path=run/'native_command_inputs.json';native=read(native_path)
    default=np.asarray(native['default_joint_position_rad']);limits=np.asarray(native['joint_limits_rad'])
    state=model.state_from_public(public[args.tick],default)
    commands=limits[:,0]+.5*(actions+1.)*np.diff(limits).ravel()
    state.previous_command_target=commands[args.tick-1].copy()
    geometry_info=read(run/'geometry.json');root_id=geometry_info['body_names'].index('base')
    poses_path=run/'all_first_episodes.npz'
    with np.load(poses_path,allow_pickle=False) as data:
        pose=data['link_poses'][args.tick//2,3,root_id];valid=data['valid_first_episode'][args.tick//2,3]
    errors=dict(position=float(np.max(abs(state.q[:3]-pose[:3]))),
                rotation=float(np.max(abs(state.root_rotation_world-rotation_wxyz(pose[3:])))))
    if not valid or max(errors.values())>2e-5:raise ValueError('Query does not match recorded native pose')
    history=RecordedNonfootContactHistory(run)
    outcome=next(r for r in read(run/'episodes.json') if r['source_training_index']==3)
    if (history.seen(3,args.tick,public[args.tick,775]) or outcome['release_time_s'] is None
            or outcome['release_time_s']>=args.tick*.01 or public[:args.tick+1,773].max()<=.5):
        raise ValueError('Query must follow observed hit/release and precede recorded original nonfoot fault')
    geometry_path=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry=AuthoredGroundClearance(geometry_path,model,margin_m=0.)
    distances,_=geometry.evaluate(state.q);nearest=int(np.argmin(distances))
    if distances[nearest]<-1e-6:raise ValueError('Initial modeled nonfoot overlap')
    recovery_path=ROOT/'configs/evaluation_protocol_v2.json';protocol=read(recovery_path)['recovery']
    spheres=read(args.sphere_audit)['spheres']
    sources=[Path(__file__),args.bundle/'manifest.json',archive,args.validation,args.warm_plan,args.sphere_audit,
        native_path,trace_manifest_path,trace_path,poses_path,run/'geometry.json',run/'episodes.json',
        run/'contact_telemetry_metadata.json',run/'first_nonfoot_contact_details.json',geometry_path,protocol_path,recovery_path,model.path]
    sources += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    hashes={str(p.resolve()):sha(p) for p in sources};results=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for mode in ['recorded_student','hold_actual_previous','recorded_stand_legs_hold_arm','existing_terminal_feedback',
                 'stand_legs_measured_arm','terminal_legs_measured_arm']:
        coarse=rollout(controller,state,feedback,dict(default=default,commands=commands),mode,
            spheres,geometry,protocol,tick=args.tick,substeps=args.screen_substeps)
        fine=(rollout(controller,state,feedback,dict(default=default,commands=commands),mode,
            spheres,geometry,protocol,tick=args.tick,substeps=160)
              if coarse['accepted'] and args.screen_substeps!=160 else None)
        results.append(dict(reference=mode,coarse=coarse,fine=fine))
        print(json.dumps(plain(dict(reference=mode,accepted=coarse['accepted'],steps=coarse['completed_model_steps'],
            failure=coarse['failure'],fine_accepted=None if fine is None else fine['accepted']))),flush=True)
        args.output.with_suffix('.partial.json').write_text(json.dumps(plain(results),allow_nan=False)+'\n')
    for p,digest in hashes.items():
        if sha(p)!=digest:raise ValueError('Changed early query source '+p)
    result=dict(scope=__doc__,case=3,tick=args.tick,initial_q=state.q,initial_v=state.v,
        initial_contacts=state.contacts,actual_previous_command=state.previous_command_target,
        recorded_pose_error=errors,initial_nonfoot_gap_m=float(distances[nearest]),
        nearest_nonfoot_body=geometry.shapes[nearest]['body'],results=results,source_sha256=hashes,
        native_execution=False,valid_expert_labels=False,new_native_successes=None,
        limitations=['Recorded student future commands are an offline comparator only.',
            'These simple references seed subsequent planning; failure does not prove unreachability.',
            'Head force is not observed in this archive; geometry alone cannot certify native legality.',
            'Any accepted reference still requires full native ground-start execution and independent audit.'])
    args.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
