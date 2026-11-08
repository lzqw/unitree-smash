"""Audit actual sequence dispatch and model error, separate from recovery score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('/home/autodrive/LZQW/unitree_smash_contact_recovery');sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve native evidence')
    read=lambda name:json.loads((a.run/name).read_text())
    job=read('job.json');inputs=read('INPUTS.json');metrics=read('metrics.json')
    result=read('sequence_native_result.json');manifest=json.loads((a.bundle/'manifest.json').read_text())
    baseline=Path(manifest['baseline']);case=result['case'];start=result['start_tick'];key=f'case_{case:03d}'
    row=next(r for r in manifest['rows'] if r['case']==case)
    steps=row['horizon_steps']
    if type(steps) is not int or not 1<=steps<=800-start:raise ValueError('Invalid declared horizon')
    expected_samples=min(steps+1,800-start)
    if (job['status']!='SUCCEEDED' or job['exit_code']!=0 or metrics['numerical_failures'] or
            metrics['actual_control_env_steps']!=102400 or inputs['training_indices']!=list(range(128)) or
            inputs['policy_sha256']!=manifest['policy_sha256'] or not result['query_match']['passed']):
        raise ValueError('Require completed original128 query-matched native execution')
    if result.get('early_termination_tick') is not None:
        rejection=dict(passed_dispatch_integrity=False,native_execution=True,complete5=metrics['complete5'],
            reason='Selected original episode ended before the planned feedback horizon; negative result retained.',
            early_termination_tick=result['early_termination_tick'],applied_steps=result['applied_steps'],
            planned_steps=steps,valid_expert_labels=False,teacher_off_learning_result=False,
            source_sha256={str(a.run/name):sha(a.run/name) for name in ['job.json','INPUTS.json','metrics.json','sequence_native_result.json']})
        a.output.write_text(json.dumps(rejection,indent=2)+'\n')
        raise SystemExit(2)
    if (result['applied_steps']!=steps or inputs['landing_mpc_cases'] is not None or
            inputs['learned_recovery_command_head'] is not None or inputs['reactive_vhsip_cases'] is not None):
        raise ValueError('Unexpected sequence length or extra teacher')
    archive=a.bundle/manifest['bundle_name']
    if sha(archive)!=manifest['bundle_sha256'] or result['bundle_sha256']!=manifest['bundle_sha256']:
        raise ValueError('Different planned sequence')
    with np.load(archive,allow_pickle=False) as z:planned=z[key+'_commands_rad'].copy()
    with np.load(a.run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'].copy();final=z['final'].copy();valid=z['first_episode'].copy()
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    if student.shape!=(800,128,18) or final.shape!=student.shape or valid.shape!=(800,128):
        raise ValueError('Incomplete full-episode dispatch recording')
    others=np.arange(128)!=case
    if not np.array_equal(student[:,others],final[:,others]) or not np.array_equal(student[:start],final[:start]):
        raise ValueError('Unexpected command override outside selected continuation')
    limits=np.asarray(manifest['configuration']['interface']['limits'])
    targets=limits[:,0]+.5*(final[:,case]+1.)*np.diff(limits).ravel()
    from scripts.check_native_foot_space_mpc_v1 import verify_feedback_dispatch
    feedback_check=verify_feedback_dispatch(a.run,a.bundle)
    command_error=feedback_check['maximum_target_error_rad']
    per_case=[];files=[];recording=read('fixed_policy_traces/manifest.json')
    old_manifest=json.loads((baseline/'fixed_policy_traces/manifest.json').read_text())
    old_entries={r['source_training_index']:r for r in old_manifest['episodes']}
    for entry in sorted(recording['episodes'],key=lambda x:x['source_training_index']):
        lane=entry['env_id'];n=entry['samples']
        if lane!=entry['source_training_index']:raise ValueError('Changed original case/lane mapping')
        path=a.run/'fixed_policy_traces'/entry['data_file'];old=baseline/'fixed_policy_traces'/entry['data_file']
        files.extend([path,old])
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed actual trace')
        if sha(old)!=old_entries[lane]['data_sha256']:raise ValueError('Changed original baseline trace')
        with np.load(path,allow_pickle=False) as z:
            actual=z['dispatched_actions'].copy();obs=unpack_public(z['packed_public'])
        if not np.array_equal(actual,final[:n,lane]) or not valid[:n,lane].all() or valid[n:,lane].any():
            raise ValueError('Actual trace differs from dispatched action or episode mask')
        with np.load(old,allow_pickle=False) as z:
            ba=z['dispatched_actions'];bp=unpack_public(z['packed_public']);common=min(start,n,len(ba)) if lane==case else min(n,len(ba))
            if lane!=case and n!=len(ba):raise ValueError('Untouched first episode length changed')
            per_case.append(dict(case=lane,prefix_samples=common,
                prefix_public_max_error=float(np.max(abs(obs[:common]-bp[:common]))),
                prefix_action_max_error=float(np.max(abs(actual[:common]-ba[:common])))))
    ticks=np.array([r['control_tick'] for r in result['records']]);n=len(ticks)
    if not np.array_equal(ticks,np.arange(start,start+n)) or n<expected_samples:
        raise ValueError('Missing actual planned horizon or subsequent response')
    if not np.array_equal(trace['final_action'],final[ticks,case]):raise ValueError('Selected command telemetry differs')
    rate=float(np.max(abs(np.diff(targets[start-1:start+n,12:],axis=0))))
    if rate>.100001:raise ValueError('Actual arm command history rate violation')
    errors=[r for r in result['records'] if r['prediction_error'] is not None]
    if len(errors)!=expected_samples:raise ValueError('Missing predicted/actual horizon samples')
    if (errors[-1]['offset']!=expected_samples-1 or
            result['prediction_endpoint_observed']!=(start+steps<800)):
        raise ValueError('Invalid terminal observation provenance')
    event=RecordedNonfootContactHistory(a.run);old_event=RecordedNonfootContactHistory(baseline)
    def event_time(history):
        value=history.first_frame[case]*history.dt
        return None if not np.isfinite(value) else float(value)
    outcome=next(r for r in metrics['complete5_case_outcomes'] if r['training_index']==case)
    prefix_ok=all(r['prefix_public_max_error']<=5e-4 and r['prefix_action_max_error']<=5e-6 for r in per_case)
    report=dict(scope=__doc__,passed_dispatch_integrity=prefix_ok,
        native_execution=True,model_accuracy_not_implied_by_dispatch_pass=True,
        case=case,start_tick=start,planned_steps=steps,recorded_followup_steps=n,
        planned_command_mapping_error_rad=command_error,
        measured_state_feedback_dispatch=feedback_check,online_mpc=True,
        command_reference='Feedback recomputed from actual state; nominal sequence is not an executed-action label',maximum_arm_command_step_rad=rate,
        untouched_lanes_equal_current_student=True,all_lane_original_prefix_equal=prefix_ok,
        per_case_prefix=per_case,query_match=result['query_match'],
        prediction_endpoint_observed=result['prediction_endpoint_observed'],
        last_observed_offset=errors[-1]['offset'],
        last_observed_horizon_error=errors[-1]['prediction_error'],
        horizon_error_max={k:max(r['prediction_error'][k] for r in errors) for k in errors[0]['prediction_error'] if k!='contacts_equal'},
        predicted_contact_mismatch_ticks=sum(not r['prediction_error']['contacts_equal'] for r in errors),
        selected_case_outcome=outcome,selected_first_nonfoot_contact_s=event_time(event),
        baseline_first_nonfoot_contact_s=event_time(old_event),
        complete5=metrics['complete5'],conditional_recovery_count=metrics['conditional_recovery_count'],
        valid_expert_labels=False,teacher_off_learning_result=False,sustained_mpc_feedback_implemented=False)
    files += [Path(__file__),ROOT/'scripts/check_foot_space_dispatch_v1.py',
        ROOT/'scripts/native_foot_space_recovery_v1.py',archive,a.bundle/'manifest.json']+[a.run/x for x in [
        'job.json','INPUTS.json','metrics.json','sequence_native_result.json','sequence_dispatch.npz',
        'sequence_native_trace.npz','fixed_policy_traces/manifest.json','first_nonfoot_contact_details.json']]
    report['source_sha256']={str(f.resolve()):sha(f) for f in files}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed_dispatch_integrity','case','last_observed_horizon_error','selected_case_outcome','complete5']}))
    if not prefix_ok:raise SystemExit(2)

if __name__=='__main__':main()
