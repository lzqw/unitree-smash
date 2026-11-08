"""Audit the declared 26-case timed-feedback trial and all untouched cases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
from scripts.native_timed_recovery_family_v1 import CASES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run','bundle','output']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Fresh audit required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    m=read(a.bundle/'manifest.json');result=read(a.run/'sequence_native_result.json')
    job=read(a.run/'job.json');inputs=read(a.run/'INPUTS.json');metrics=read(a.run/'metrics.json')
    if (job['status']!='SUCCEEDED' or metrics['actual_control_env_steps']!=102400 or metrics['numerical_failures']
            or inputs['training_indices']!=list(range(128)) or inputs['policy_sha256']!=m['policy_sha256']
            or not result['query_match']['passed'] or m['model_accepted'] or result['model_accepted']):
        raise ValueError('Completed original128 explicitly experimental run required')
    case=m['case'];baseline=Path(m['baseline']);start=280
    with np.load(a.run/'sequence_dispatch.npz',allow_pickle=False) as z:
        student=z['student'];final=z['final'];valid=z['first_episode']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    if student.shape!=(800,128,18) or final.shape!=student.shape:raise ValueError('Incomplete commands')
    family=read(a.run/'recovery_family_result.json')
    if family['cases']!=list(CASES) or case!=85:raise ValueError('Declared family differs')
    others=~np.isin(np.arange(128),CASES)
    if not np.array_equal(student[:,others],final[:,others]) or not np.array_equal(student[:start],final[:start]):
        raise ValueError('Undeclared action override')
    ticks=np.array([r['tick'] for r in result['records']],int);n=len(ticks)
    if not np.array_equal(ticks,np.arange(start,start+n)) or n!=result['applied_steps']:
        raise ValueError('Missing actual controller records')
    if not np.array_equal(trace['final_action'],final[ticks,case]):raise ValueError('Trace/dispatch mismatch')
    if not np.array_equal(valid[start:,case],np.arange(520)<n):raise ValueError('Episode termination mismatch')
    limits=np.asarray(m['configuration']['interface']['limits'])
    decoded=limits[:,0]+.5*(trace['final_action']+1)*np.diff(limits).ravel()
    mapping=float(np.max(abs(decoded-trace['requested_target_rad']),initial=0.))
    history=limits[:,0]+.5*(final[start-1,case]+1)*np.diff(limits).ravel()
    chain=np.vstack([history,decoded]);slew=float(np.max(abs(np.diff(chain[:,12:],axis=0)),initial=0.))
    if mapping>1e-6 or slew>.100001 or np.any(decoded<limits[:,0]-1e-6) or np.any(decoded>limits[:,1]+1e-6):
        raise ValueError('Actual command interface violation')
    previous=np.vstack([history,decoded[:-1]])
    if not np.allclose(previous,trace['previous_target_rad'],atol=1e-12,rtol=0):raise ValueError('Command history mismatch')
    with np.load(a.run/'recovery_family_trace.npz',allow_pickle=False) as z:extra={k:z[k].copy() for k in z.files}
    extra_checks=[]
    for lane in CASES:
        if lane==85:continue
        selected=extra['case']==lane;tick=extra['tick'][selected];count=len(tick)
        if (not np.array_equal(tick,np.arange(280,280+count))
                or not np.array_equal(valid[280:,lane],np.arange(520)<count)):
            raise ValueError('Missing family dispatch or incorrect termination')
        if not np.array_equal(extra['final_action'][selected],final[tick,lane]):raise ValueError('Family dispatch differs')
        actual=limits[:,0]+.5*(final[tick,lane]+1)*np.diff(limits).ravel()
        prior=limits[:,0]+.5*(final[279,lane]+1)*np.diff(limits).ravel()
        histories=np.vstack([prior,actual[:-1]])
        err=float(np.max(abs(actual-extra['requested_target_rad'][selected]),initial=0.))
        rate=float(np.max(abs(actual[:,12:]-histories[:,12:]),initial=0.))
        if (err>1e-6 or rate>.100001 or not np.allclose(histories,extra['previous_target_rad'][selected],atol=1e-12,rtol=0)
                or np.any(actual<limits[:,0]-1e-6) or np.any(actual>limits[:,1]+1e-6)):
            raise ValueError('Family interface/history violation')
        extra_checks.append(dict(case=lane,steps=count,mapping_error_rad=err,maximum_arm_step_rad=rate))
    recording=read(a.run/'fixed_policy_traces/manifest.json');old=read(baseline/'fixed_policy_traces/manifest.json')
    old_entries={r['source_training_index']:r for r in old['episodes']};per_case=[];pins={}
    if len(recording['episodes'])!=128:raise ValueError('Full original panel required')
    for entry in recording['episodes']:
        lane=entry['env_id'];p=a.run/'fixed_policy_traces'/entry['data_file'];b=baseline/'fixed_policy_traces'/entry['data_file']
        if lane!=entry['source_training_index'] or sha(p)!=entry['data_sha256'] or sha(b)!=old_entries[lane]['data_sha256']:
            raise ValueError('Trace identity/hash changed')
        pins[str(p.resolve())]=sha(p);pins[str(b.resolve())]=sha(b)
        with np.load(p,allow_pickle=False) as z:acts=z['dispatched_actions'].copy();obs=unpack_public(z['packed_public'])
        with np.load(b,allow_pickle=False) as z:old_acts=z['dispatched_actions'].copy();old_obs=unpack_public(z['packed_public'])
        if not np.array_equal(acts,final[:len(acts),lane]):raise ValueError('Per-case executed actions differ')
        common=min(start,len(acts),len(old_acts))
        prefix=bool(np.array_equal(acts[:common],old_acts[:common]) and np.array_equal(obs[:common],old_obs[:common]))
        exact=bool(acts.shape==old_acts.shape and np.array_equal(acts,old_acts) and np.array_equal(obs,old_obs))
        per_case.append(dict(case=lane,prefix_exact=prefix,whole_trace_exact=exact,
            current_samples=len(acts),baseline_samples=len(old_acts)))
    outcome=next(r for r in metrics['complete5_case_outcomes'] if r['training_index']==case)
    history=RecordedNonfootContactHistory(a.run);t=history.first_frame[case]*history.dt
    for p in [a.bundle/'manifest.json',a.run/'job.json',a.run/'INPUTS.json',a.run/'metrics.json',
            a.run/'recovery_family_result.json',a.run/'recovery_family_trace.npz',a.run/'sequence_native_result.json',a.run/'sequence_dispatch.npz',a.run/'sequence_native_trace.npz',Path(__file__).resolve()]:
        pins[str(p.resolve())]=sha(p)
    previous_metrics=read(baseline/'metrics.json')
    old_success={r['training_index'] for r in previous_metrics['complete5_case_outcomes'] if r['complete5']}
    new_success={r['training_index'] for r in metrics['complete5_case_outcomes'] if r['complete5']}
    old_ball={r['training_index'] for r in previous_metrics['complete5_case_outcomes'] if r['ball5_except_recovery']}
    new_ball={r['training_index'] for r in metrics['complete5_case_outcomes'] if r['ball5_except_recovery']}
    audit=dict(scope=__doc__,case=case,family_cases=list(CASES),family_dispatch_checks=extra_checks,
        complete5_changes=dict(retained=sorted(old_success&new_success),lost=sorted(old_success-new_success),gained=sorted(new_success-old_success)),
        ball5_changes=dict(retained=sorted(old_ball&new_ball),lost=sorted(old_ball-new_ball),gained=sorted(new_ball-old_ball)),
        family_outcomes=[r for r in metrics['complete5_case_outcomes'] if r['training_index'] in CASES],
        passed_dispatch_integrity=all(r['prefix_exact'] for r in per_case),
        actual_mapping_error_rad=mapping,maximum_arm_command_step_rad=slew,
        strict_untouched_isolation_passed=all(r['whole_trace_exact'] for r in per_case if r['case'] not in CASES),
        differing_untouched=[r for r in per_case if r['case'] not in CASES and not r['whole_trace_exact']],
        per_case=per_case,complete5=metrics['complete5'],selected_case_outcome=outcome,
        selected_first_scored_nonfoot_s=float(t) if np.isfinite(t) else None,
        early_termination_tick=result['early_termination_tick'],source_sha256=pins,
        model_accepted=False,automatic_promotion=False,valid_expert_labels=False,
        note='Dispatch consistency does not prove model accuracy, all-body clearance or expert quality.')
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    print(json.dumps({k:audit[k] for k in ['case','complete5','selected_case_outcome','passed_dispatch_integrity','strict_untouched_isolation_passed']}))


if __name__=='__main__':main()
