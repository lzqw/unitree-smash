"""Check dispatch and actual-versus-predicted motion in an immutable native prefix.

Incomplete runs never earn recovery/admission labels. The comparison uses the
plan active for the preceding executed command, including at replan boundaries.
It does not confuse an old nominal trajectory with the current prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior prefix audit')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prefix=read(args.prefix);folder=args.prefix.parent
    if (prefix['schema']!='diagnostic_recovery_prefix_v1' or prefix['case']!=3
            or prefix['start_tick']!=460 or prefix['full_run_admission'] or prefix['valid_expert_labels']):
        raise ValueError('Original diagnostic prefix required')
    trace_path=folder/prefix['trace_file']
    if sha(trace_path)!=prefix['trace_sha256']: raise ValueError('Changed prefix trace')
    with np.load(trace_path,allow_pickle=False) as data: trace={key:data[key].copy() for key in data.files}
    n=prefix['executed_recovery_records'];records=prefix['records']
    if (n!=len(records) or not 1<=n<=340 or prefix['completed_control_ticks']!=460+n
            or not np.array_equal(trace['control_tick'],np.arange(460,460+n))
            or any(len(value)!=n or not np.isfinite(value).all() for value in trace.values())):
        raise ValueError('Discontinuous/incomplete executed prefix')
    model=LoadedPinocchioModelV1();manifest_path=args.bundle/'manifest.json';manifest=read(manifest_path)
    limits=np.asarray(manifest['configuration']['interface']['limits'])
    decoded=limits[:,0]+.5*(trace['final_action']+1.)*np.diff(limits).ravel()
    decode_error=float(np.max(abs(decoded-trace['final_target_rad'])))
    if decode_error>1e-7: raise ValueError('Final target is not decoded actual action')
    sources=[Path(__file__),args.prefix,trace_path,model.path,manifest_path]
    plans={}
    for pointer in prefix['planning_records']:
        path=folder/pointer['file'];sources.append(path)
        if sha(path)!=pointer['sha256']: raise ValueError('Changed planning record')
        plan=read(path)
        if plan['tick']!=pointer['tick'] or plan['tick'] in plans: raise ValueError('Ambiguous plan identity')
        plans[plan['tick']]=plan
    active=None;last_interval_plan=None;errors=[];request_error=0.;teacher=0;fallback=0
    for i,record in enumerate(records):
        tick=460+i
        if record['control_tick']!=tick: raise ValueError('Wrong executed clock')
        if i and last_interval_plan is not None:
            elapsed=tick-last_interval_plan['tick']
            predicted=last_interval_plan['decision']['predicted_states'][elapsed-1]
            q=np.asarray(predicted['q']);v=np.asarray(predicted['v'])
            dq=pin.difference(model.model,q,trace['q'][i])
            real_R=pin.Quaternion(trace['q'][i,3:7]).matrix();pred_R=pin.Quaternion(q[3:7]).matrix()
            errors.append(dict(control_tick=tick,plan_tick=last_interval_plan['tick'],prediction_steps=elapsed,
                joint_position_max_rad=float(np.max(abs(dq[model.v_indices]))),
                root_position_m=float(np.linalg.norm(trace['q'][i,:3]-q[:3])),
                root_attitude_rad=float(np.linalg.norm(dq[3:6])),
                joint_velocity_max_rad_s=float(np.max(abs(trace['v'][i,model.v_indices]-v[model.v_indices]))),
                root_linear_velocity_world_m_s=float(np.linalg.norm(real_R@trace['v'][i,:3]-pred_R@v[:3])),
                root_angular_velocity_world_rad_s=float(np.linalg.norm(real_R@trace['v'][i,3:6]-pred_R@v[3:6])),
                contact_flags_equal=bool(np.array_equal(trace['contacts'][i],predicted['contacts']))))
        if tick in plans:
            plan=plans[tick]
            for field,name in [('observed_q','q'),('observed_v','v'),('observed_contacts','contacts'),
                               ('actual_previous_target_rad','previous_target_rad')]:
                if not np.array_equal(plan[field],trace[name][i]): raise ValueError('Stale actual query '+field)
            active=plan if plan['decision']['accepted'] else None
        if record['mode']=='frictional_receding_support_sequence':
            if active is None or not 0<=tick-active['tick']<10: raise ValueError('Missing or stale active plan')
            expected=np.asarray(active['decision']['commands'])[tick-active['tick']];teacher+=1
            last_interval_plan=active
        elif record['mode']=='planning_failed_student_fallback_not_certified':
            if active is not None: raise ValueError('Fallback conflicts with accepted plan')
            expected=limits[:,0]+.5*(trace['student_action'][i]+1.)*np.diff(limits).ravel()
            expected[12:]=np.clip(expected[12:],trace['previous_target_rad'][i,12:]-.1,trace['previous_target_rad'][i,12:]+.1)
            fallback+=1;last_interval_plan=None
        else: raise ValueError('Unexpected native control mode')
        error=float(np.max(abs(expected-trace['final_target_rad'][i])));request_error=max(request_error,error)
        if error>1e-6 or np.max(abs(expected-trace['requested_target_rad'][i]))>1e-8:
            raise ValueError('Selected command differs from dispatch')
        if i and np.max(abs(trace['previous_target_rad'][i]-trace['final_target_rad'][i-1]))>1e-6:
            raise ValueError('Actual command history differs')
    summary={}
    fields=['joint_position_max_rad','root_position_m','root_attitude_rad','joint_velocity_max_rad_s',
            'root_linear_velocity_world_m_s','root_angular_velocity_world_rad_s']
    for name,subset in [('all_executed_intervals',errors),('one_step',[r for r in errors if r['prediction_steps']==1])]:
        summary[name]=dict(samples=len(subset),contact_flag_disagreements=sum(not r['contact_flags_equal'] for r in subset),
            errors={field:dict(median=float(np.median([r[field] for r in subset])),maximum=max(r[field] for r in subset))
                    for field in fields} if subset else {})
    result=dict(scope=__doc__,executed_steps=n,teacher_steps=teacher,unverified_fallback_steps=fallback,
        accepted_plans=sum(p['decision']['accepted'] for p in plans.values()),
        maximum_request_to_dispatch_error_rad=request_error,maximum_decode_error_rad=decode_error,
        prediction_summary=summary,prediction_records=errors,
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        full_run_admission=False,new_complete_successes=None,valid_expert_labels=False,
        limitations=['The live run is unfinished; no full-case isolation or independent full-model audit is inferred.',
            'Native force-hysteresis flags and model positive-impulse flags have different definitions.',
            'Error correlation does not identify missing head contact, friction or actuator dynamics as the unique cause.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['teacher_steps','accepted_plans','prediction_summary']}))


if __name__=='__main__':main()
