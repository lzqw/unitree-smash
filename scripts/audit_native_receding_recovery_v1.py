"""Audit feedback decisions against actual dispatch, observations and outcomes."""
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
from scripts.audit_microstep_recovery_v1 import characterize
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_dispatch(result,trace,student,final,limits):
    """Check feedback semantics without trusting accepted flags or action labels."""
    case=result['case'];start=result['start_tick'];steps=result['planned_steps'];records=result['records']
    if start<1 or start+steps!=800 or len(records)!=steps:raise ValueError('Full remaining episode required')
    if student.shape!=(800,128,18) or final.shape!=student.shape:raise ValueError('Original panel required')
    if not np.isfinite(student).all() or not np.isfinite(final).all() or np.max(abs(final))>1.0000001:
        raise ValueError('Invalid actual action')
    others=np.arange(128)!=case
    if not np.array_equal(student[:,others],final[:,others]) or not np.array_equal(student[:start],final[:start]):
        raise ValueError('Override outside declared recovery window')
    decode=lambda values:limits[:,0]+.5*(values+1.)*np.diff(limits).ravel()
    targets=decode(final[:,case]);raw=decode(student[:,case]);plans=result['planning_records']
    ticks=[p['tick'] for p in plans]
    if not ticks or ticks[0]!=start or ticks!=sorted(set(ticks)) or ticks[-1]>=800:
        raise ValueError('Missing, duplicate or unordered replans')
    by_tick={p['tick']:p for p in plans};active=None;failed=False;teacher=0;fallback=0;maximum=0.
    if not np.array_equal(trace['final_action'],final[start:,case]):raise ValueError('Trace differs from final dispatch')
    for i,record in enumerate(records):
        tick=start+i
        if record['control_tick']!=tick or record['offset']!=i:raise ValueError('Discontinuous observed control clock')
        if not np.allclose(trace['previous_target_rad'][i],targets[tick-1],atol=1e-6,rtol=0):
            raise ValueError('Prior target is not actual dispatch history')
        if not np.array_equal(trace['student_action'][i],student[tick,case]):raise ValueError('Wrong raw student action')
        if not np.allclose(trace['final_target_rad'][i],targets[tick],atol=1e-7,rtol=0):
            raise ValueError('Wrong final command telemetry')
        if tick in by_tick:
            if failed:raise ValueError('Replanning after latched failure is not declared behavior')
            plan=by_tick[tick]
            for name,field in [('observed_q','q'),('observed_v','v'),('observed_contacts','contacts'),
                    ('measured_foot_force_world_n','foot_force_world_n'),('actual_previous_target_rad','previous_target_rad')]:
                if not np.array_equal(np.asarray(plan[name]),trace[field][i]):raise ValueError('Planner used stale '+name)
            changed=active is not None and not np.array_equal(trace['contacts'][i],active['observed_contacts'])
            if bool(plan['contact_changed'])!=changed:raise ValueError('Contact-change reason differs')
            decision=plan['decision']
            if decision['accepted']:
                commands=np.asarray(decision['commands'])
                if commands.shape!=(40,18) or not np.isfinite(commands).all():raise ValueError('Invalid accepted horizon')
                for name,field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('previous_command','previous_target_rad')]:
                    if not np.array_equal(np.asarray(decision[name]),trace[field][i]):raise ValueError('Search used stale '+name)
                active=plan
            else:
                if decision['commands'] is not None:raise ValueError('Failed search returned an executable command')
                failed=True
        if failed:
            expected=raw[tick].copy();expected[12:]=np.clip(expected[12:],targets[tick-1,12:]-.1,targets[tick-1,12:]+.1)
            if record['mode']!='planning_failed_student_fallback_not_certified':raise ValueError('Fallback falsely labelled teacher')
            fallback+=1
        else:
            if active is None or tick-active['tick']>=10:raise ValueError('Stale plan exceeds replan interval')
            if not np.array_equal(trace['contacts'][i],active['observed_contacts']):raise ValueError('Contact change ignored')
            expected=np.asarray(active['decision']['commands'])[tick-active['tick']]
            if record['mode']!='receding_support_sequence':raise ValueError('Wrong teacher mode')
            teacher+=1
        error=float(np.max(abs(expected-targets[tick])));maximum=max(maximum,error)
        if error>1e-6 or np.max(abs(trace['requested_target_rad'][i]-expected))>1e-8:
            raise ValueError('Planner/fallback request was not the actual command')
    rate=float(np.max(abs(np.diff(targets[start-1:,12:],axis=0))))
    if rate>.100001:raise ValueError('Original actual arm slew violated')
    if (teacher!=result['teacher_steps'] or fallback!=result['unverified_student_fallback_steps']
            or bool(result['planning_failed'])!=failed):raise ValueError('Incorrect intervention counts')
    return dict(teacher_steps=teacher,unverified_fallback_steps=fallback,replans=len(plans),
        accepted_replans=sum(bool(p['decision']['accepted']) for p in plans),
        maximum_command_mapping_error_rad=maximum,maximum_arm_command_step_rad=rate,
        untouched_lanes_equal_current_student=True,actual_observations_and_command_history_verified=True)


def state_at(model,trace,index):
    q=trace['q'][index].copy();v=trace['v'][index].copy()
    state=PublicState(q=q,v=v,root_rotation_world=pin.Quaternion(q[3:7]).matrix(),root_position_world=q[:3].copy(),
        joint_position=q[model.q_indices].copy(),joint_velocity=v[model.v_indices].copy(),
        contacts=trace['contacts'][index].copy(),rotation_projection_error=0.)
    state.previous_command_target=trace['previous_target_rad'][index].copy();return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','output']:p.add_argument('--'+name,type=Path,required=True)
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
            not result['sustained_mpc_feedback_implemented'] or result['valid_expert_labels']):
        raise ValueError('Require completed unchanged native panel with raw diagnostic labels')
    for key in ['landing_mpc_cases','learned_recovery_command_head','reactive_vhsip_cases','expert_bank']:
        if inputs[key] is not None:raise ValueError('Unexpected extra teacher '+key)
    if sha(a.bundle/manifest['bundle_name'])!=manifest['bundle_sha256'] or result['bundle_sha256']!=manifest['bundle_sha256']:
        raise ValueError('Changed initial sequence')
    with np.load(a.run/'sequence_dispatch.npz',allow_pickle=False) as z:student=z['student'];final=z['final'];valid=z['first_episode']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    limits=np.asarray(manifest['configuration']['interface']['limits']);dispatch=audit_dispatch(result,trace,student,final,limits)
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
    for j,plan in enumerate(result['planning_records']):
        if not plan['decision']['accepted']:continue
        index=plan['tick']-start;state=state_at(c.qp.model,trace,index);commands=np.asarray(plan['decision']['commands'])
        reference=limits[:,0]+.5*(student[plan['tick'],case]+1.)*np.diff(limits).ravel()
        checked,_=characterize(c,state,commands,reference)
        if not checked['accepted']:raise ValueError('Stored accepted plan fails independent replay')
        prediction=rollout_for_search(c,state,commands,reference)
        end=result['planning_records'][j+1]['tick'] if j+1<len(result['planning_records']) else 800
        errors=[]
        for offset in range(1,min(end-plan['tick'],len(trace['q'])-1-index)+1):
            expected=prediction['profile'][offset-1]['following'];actual=index+offset
            delta=pin.difference(c.qp.model.model,expected.q,trace['q'][actual])
            errors.append(dict(tick=plan['tick']+offset,configuration_tangent_max=float(np.max(abs(delta))),
                root_position_m=float(np.linalg.norm(expected.q[:3]-trace['q'][actual,:3])),
                root_attitude_rad=float(np.linalg.norm(delta[3:6])),velocity_max=float(np.max(abs(expected.v-trace['v'][actual]))),
                contacts_equal=bool(np.array_equal(expected.contacts,trace['contacts'][actual]))))
        verified.append(dict(tick=plan['tick'],independently_accepted=True,verification=checked,
            actual_executed_prediction_errors=errors,wall_seconds=plan['decision']['wall_seconds']))
    event=RecordedNonfootContactHistory(a.run);old_event=RecordedNonfootContactHistory(baseline)
    time_of=lambda e:None if not np.isfinite(e.first_frame[case]) else float(e.first_frame[case]*e.dt)
    outcome=next(r for r in metrics['complete5_case_outcomes'] if r['training_index']==case)
    report=dict(scope=__doc__,passed_dispatch_integrity=equivalent,dispatch=dispatch,case=case,start_tick=start,
        native_execution=True,sustained_mpc_feedback_implemented=True,valid_expert_labels=False,
        all_other_lanes_and_selected_prefix_equal_baseline=equivalent,per_case_comparison=comparisons,
        independent_replan_checks=verified,selected_case_outcome=outcome,complete5=metrics['complete5'],
        conditional_recovery_count=metrics['conditional_recovery_count'],
        selected_first_nonfoot_contact_s=time_of(event),baseline_first_nonfoot_contact_s=time_of(old_event),
        teacher_off_learning_result=False,automatic_promotion=False)
    files += [Path(__file__),geometry,a.bundle/'manifest.json',a.bundle/manifest['bundle_name']]
    files += [a.run/n for n in ['job.json','INPUTS.json','metrics.json','sequence_native_result.json',
        'sequence_dispatch.npz','sequence_native_trace.npz','fixed_policy_traces/manifest.json','first_nonfoot_contact_details.json']]
    report['source_sha256']={str(f.resolve()):sha(f) for f in files}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['passed_dispatch_integrity','case','dispatch','selected_case_outcome','complete5']}))
    if not equivalent:raise SystemExit(2)


if __name__=='__main__':main()
