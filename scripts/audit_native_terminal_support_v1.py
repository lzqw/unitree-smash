"""Audit persistent terminal feedback, actual dispatch and the complete predicted trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.native_terminal_support_v1 import load_verified_policy
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_dispatch(result,trace,student,final,limits,policy):
    case=result['case'];start=result['start_tick'];steps=result['planned_steps'];records=result['records']
    if start!=460 or steps!=340 or len(records)!=340:raise ValueError('Complete original episode required')
    if student.shape!=(800,128,18) or final.shape!=student.shape:raise ValueError('Original panel required')
    if not np.isfinite(final).all() or np.max(abs(final))>1.0000001:raise ValueError('Invalid final commands')
    others=np.arange(128)!=case
    if not np.array_equal(student[:,others],final[:,others]) or not np.array_equal(student[:start],final[:start]):
        raise ValueError('Unexpected intervention outside declared interval')
    decode=lambda x:limits[:,0]+.5*(x+1.)*np.diff(limits).ravel()
    targets=decode(final[:,case]);maximum=0.
    if not np.array_equal(trace['final_action'],final[start:,case]):raise ValueError('Trace differs from dispatch')
    for i,record in enumerate(records):
        tick=start+i
        if record['control_tick']!=tick or record['offset']!=i:raise ValueError('Discontinuous control clock')
        if not np.array_equal(trace['student_action'][i],student[tick,case]):raise ValueError('Wrong original action')
        if not np.allclose(trace['previous_target_rad'][i],targets[tick-1],atol=1e-6,rtol=0):
            raise ValueError('Wrong command history')
        state=state_at(policy.model,trace,i)
        expected,mode=policy.action(i,state,trace['previous_target_rad'][i])
        if mode!=record['mode']:raise ValueError('Wrong feedback stage')
        error=float(np.max(abs(expected-targets[tick])));maximum=max(maximum,error)
        if error>1e-6 or np.max(abs(trace['requested_target_rad'][i]-expected))>1e-8:
            raise ValueError('Feedback differs from actual request/dispatch')
        if not np.allclose(trace['final_target_rad'][i],targets[tick],atol=1e-7,rtol=0):
            raise ValueError('Wrong realized target telemetry')
    rate=float(np.max(abs(np.diff(targets[start-1:,12:],axis=0))))
    if rate>.100001 or result['teacher_steps']!=340 or result['unverified_student_fallback_steps']!=0:
        raise ValueError('Wrong slew or control coverage')
    return dict(teacher_steps=340,prefix_feedback_steps=80,terminal_feedback_steps=260,
        unverified_fallback_steps=0,maximum_command_mapping_error_rad=maximum,
        maximum_arm_command_step_rad=rate,untouched_lanes_equal_current_student=True,
        actual_observations_and_command_history_verified=True,online_replanning=False)


def state_at(model,trace,index):
    q=trace['q'][index].copy();v=trace['v'][index].copy()
    state=PublicState(q=q,v=v,root_rotation_world=pin.Quaternion(q[3:7]).matrix(),root_position_world=q[:3].copy(),
        joint_position=q[model.q_indices].copy(),joint_velocity=v[model.v_indices].copy(),
        contacts=trace['contacts'][index].copy(),rotation_projection_error=0.)
    state.previous_command_target=trace['previous_target_rad'][index].copy();return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','output','sphere-audit']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve native audit')
    read=lambda name:json.loads((a.run/name).read_text())
    job=read('job.json');inputs=read('INPUTS.json');metrics=read('metrics.json');result=read('sequence_native_result.json')
    manifest=json.loads((a.bundle/'manifest.json').read_text());baseline=Path(manifest['baseline'])
    old_inputs=json.loads((baseline/'INPUTS.json').read_text());case=result['case'];start=result['start_tick']
    for key in ['policy_sha256','protocol_sha256','training_suite','origin_precision_receipt_sha256',
            'origin_spacing_m','training_indices','reset_noise_amplitudes','ground_extent_diagnostic',
            'recovery_assistance_execution','support_config','support_disabled','residual_actor_enabled']:
        if inputs[key]!=old_inputs[key]:raise ValueError('Changed task/control contract '+key)
    old_meta=json.loads((baseline/'contact_telemetry_metadata.json').read_text());meta=read('contact_telemetry_metadata.json')
    if meta['physics_dt_s']!=old_meta['physics_dt_s']:raise ValueError('Changed physics time step')
    if (job['status']!='SUCCEEDED' or job['exit_code']!=0 or metrics['numerical_failures'] or
            metrics['actual_control_env_steps']!=102400 or inputs['training_indices']!=list(range(128)) or
            inputs['policy_sha256']!=manifest['policy_sha256'] or not result['query_match']['passed'] or
            not result['sustained_terminal_feedback_implemented'] or result['valid_expert_labels']):
        raise ValueError('Require completed unchanged native panel with raw diagnostic labels')
    for key in ['landing_mpc_cases','learned_recovery_command_head','reactive_vhsip_cases','expert_bank']:
        if inputs[key] is not None:raise ValueError('Unexpected extra teacher '+key)
    if sha(a.bundle/manifest['bundle_name'])!=manifest['bundle_sha256'] or result['bundle_sha256']!=manifest['bundle_sha256']:
        raise ValueError('Changed initial sequence')
    with np.load(a.run/'sequence_dispatch.npz',allow_pickle=False) as z:student=z['student'];final=z['final'];valid=z['first_episode']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    limits=np.asarray(manifest['configuration']['interface']['limits'])
    c=build_controller(manifest['configuration'])
    with np.load(a.bundle/manifest['bundle_name'],allow_pickle=False) as z:query=z[f'case_{case:03d}_query_public'].copy()
    policy,validation=load_verified_policy(c.qp.model,c,query,Path(inputs['frictional_warm_plan']),Path(result['validation_path']))
    if sha(result['validation_path'])!=result['validation_sha256'] or inputs['terminal_support_validation_sha256']!=result['validation_sha256']:
        raise ValueError('Different feedback validation')
    dispatch=audit_dispatch(result,trace,student,final,limits,policy)
    files=[];comparisons=[];entries=read('fixed_policy_traces/manifest.json')['episodes']
    old_entries={r['source_training_index']:r for r in json.loads((baseline/'fixed_policy_traces/manifest.json').read_text())['episodes']}
    if sorted(r['source_training_index'] for r in entries)!=list(range(128)):raise ValueError('Missing original cases')
    for entry in entries:
        lane=entry['env_id'];n=entry['samples'];path=a.run/'fixed_policy_traces'/entry['data_file']
        old=baseline/'fixed_policy_traces'/old_entries[lane]['data_file'];files += [path,old]
        if lane!=entry['source_training_index'] or sha(path)!=entry['data_sha256'] or sha(old)!=old_entries[lane]['data_sha256']:
            raise ValueError('Changed trace provenance')
        with np.load(path,allow_pickle=False) as z:actions=z['dispatched_actions'].copy();obs=unpack_public(z['packed_public'])
        if not np.array_equal(actions,final[:n,lane]) or not valid[:n,lane].all() or valid[n:,lane].any():raise ValueError('Final command was not executed')
        with np.load(old,allow_pickle=False) as z:ba=z['dispatched_actions'];bo=unpack_public(z['packed_public'])
        count=min(start,n,len(ba)) if lane==case else min(n,len(ba))
        comparisons.append(dict(case=lane,samples=count,same_episode_length=n==len(ba),
            public_max_error=float(np.max(abs(obs[:count]-bo[:count]))),
            action_max_error=float(np.max(abs(actions[:count]-ba[:count])))))
    equivalent=all(r['public_max_error']<=5e-4 and r['action_max_error']<=5e-6 and
        (r['case']==case or r['same_episode_length']) for r in comparisons)
    c=build_controller(manifest['configuration']);geometry=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    if sha(geometry)!=manifest['ground_geometry_sha256']:raise ValueError('Changed geometry')
    c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.);verified=[]
    sphere_record=json.loads(a.sphere_audit.read_text());spheres=sphere_record['spheres']
    if (sha(a.sphere_audit)!=result['sphere_audit_sha256'] or
            inputs['frictional_sphere_audit_sha256']!=result['sphere_audit_sha256'] or
            inputs['frictional_warm_plan_sha256']!=result['warm_plan_sha256'] or
            result['teacher_model']!='planned_prefix_terminal_feedback_v1' or result['online_replanning']):
        raise ValueError('Different predictor, horizon or pinned input')
    for name,digest in sphere_record['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Changed authored sphere evidence '+name)

    state=state_at(c.qp.model,trace,0)
    # The saved validation starts at an earlier native run's measured query.
    # Reproduce it exactly, then report current-run tracking separately.
    from scripts.extend_frictional_recovery_v1 import set_state
    state=set_state(state,c.qp.model,validation['initial_q'],validation['initial_v'],
        validation['initial_contacts'],validation['actual_previous_command'])
    max_q=0.;max_v=0.;errors=[]
    for offset in range(340):
        command,mode=policy.action(offset,state,state.previous_command_target)
        checked=predict_frictional_spheres(c.qp.model,c.qp.interface,state,command,spheres,geometry=c.sequence_geometry)
        d=checked['diagnostics']
        if d['nonfoot_ground_violation_m']>1e-6 or any(d[k]>3e-4 for k in
            ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm']):
            raise ValueError('Independent full model replay violates constraints')
        state=checked['state'];state.previous_command_target=command.copy()
        saved=validation['states'][offset]
        max_q=max(max_q,float(np.max(abs(state.q-np.asarray(saved['q'])))))
        max_v=max(max_v,float(np.max(abs(state.v-np.asarray(saved['v'])))))
        if not np.allclose(command,validation['commands'][offset],atol=1e-10,rtol=0):raise ValueError('Different nominal feedback command')
        if offset+1<340:
            delta=pin.difference(c.qp.model.model,state.q,trace['q'][offset+1])
            errors.append(dict(tick=start+offset+1,root_position_m=float(np.linalg.norm(delta[:3])),
                root_attitude_rad=float(np.linalg.norm(delta[3:6])),joint_max_rad=float(np.max(abs(delta[c.qp.model.v_indices])))))
    if max(max_q,max_v)>1e-10:raise ValueError('Independent predicted trajectory differs')
    verified=dict(full_model_steps=340,maximum_saved_q_error=max_q,maximum_saved_v_error=max_v,
        actual_nominal_tracking_errors=errors,
        scope='Closed-loop nominal trajectory tracking; not a prediction using the differing native commands')
    event=RecordedNonfootContactHistory(a.run);old_event=RecordedNonfootContactHistory(baseline)
    time_of=lambda e:None if not np.isfinite(e.first_frame[case]) else float(e.first_frame[case]*e.dt)
    outcome=next(r for r in metrics['complete5_case_outcomes'] if r['training_index']==case)
    report=dict(scope=__doc__,passed_dispatch_integrity=equivalent,dispatch=dispatch,case=case,start_tick=start,
        native_execution=True,sustained_mpc_feedback_implemented=False,sustained_terminal_feedback_implemented=True,valid_expert_labels=False,
        teacher_model='planned_prefix_terminal_feedback_v1',model_steps=340,
        all_other_lanes_and_selected_prefix_equal_baseline=equivalent,per_case_comparison=comparisons,
        independent_model_check=verified,selected_case_outcome=outcome,complete5=metrics['complete5'],
        conditional_recovery_count=metrics['conditional_recovery_count'],
        selected_first_nonfoot_contact_s=time_of(event),baseline_first_nonfoot_contact_s=time_of(old_event),
        teacher_off_learning_result=False,automatic_promotion=False)
    files += [Path(result['validation_path']),Path(inputs['frictional_warm_plan']),Path(__file__),geometry,a.bundle/'manifest.json',a.bundle/manifest['bundle_name'],a.sphere_audit]
    files += [a.run/n for n in ['job.json','INPUTS.json','metrics.json','sequence_native_result.json',
        'sequence_dispatch.npz','sequence_native_trace.npz','fixed_policy_traces/manifest.json','first_nonfoot_contact_details.json']]
    report['source_sha256']={str(f.resolve()):sha(f) for f in files}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['passed_dispatch_integrity','case','dispatch','selected_case_outcome','complete5']}))
    if not equivalent:raise SystemExit(2)


if __name__=='__main__':main()
