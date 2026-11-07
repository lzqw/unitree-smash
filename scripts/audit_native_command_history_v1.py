#!/usr/bin/env python3
"""Cross-check controller telemetry against recorded native dispatched actions.

This verifies sampled command mapping, not actuator dynamics or stable recovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arm-target-rate',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tolerance-rad',type=float,default=5e-6)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior audit')
    if not np.isfinite(a.tolerance_rad) or a.tolerance_rad<=0:raise ValueError('Invalid tolerance')
    read=lambda f:json.loads((a.run/f).read_text())
    job=read('job.json');inputs=read('INPUTS.json');metrics=read('metrics.json')
    if job.get('status')!='SUCCEEDED' or job.get('exit_code')!=0:raise ValueError('Native run did not finish successfully')
    if inputs['training_indices']!=list(range(128)):raise ValueError('This audit requires the original ordered TRAIN128 mapping')
    records=read('landing_mpc_records.json');native=read('native_command_inputs.json')
    limits=np.asarray(native['joint_limits_rad']);checks=[];exceptions=[];unmatched=[];cache={};hashes={}
    def hash_file(path):hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for r in records:
        if str(r.get('rejection','')).startswith('exception:'):
            exceptions.append(dict(case=r['case'],clock_s=r['clock_s'],rejection=r['rejection']));continue
        if r.get('episode_index')!=0 or 'final_dispatched_joint_target_rad' not in r:
            unmatched.append(dict(case=r['case'],clock_s=r['clock_s'],reason='missing first-episode command telemetry'));continue
        case=int(r['case'])
        if case not in cache:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';hash_file(path)
            with np.load(path) as trace:
                public=unpack_public(trace['packed_public']);actions=trace['dispatched_actions'].copy()
            ticks=np.rint(public[:,775]/.01).astype(np.int64)
            if len(np.unique(ticks))!=len(ticks):raise ValueError('Ambiguous first-episode clock')
            targets=limits[:,0]+.5*(np.clip(actions,-1.,1.)+1.)*(limits[:,1]-limits[:,0])
            cache[case]=(ticks,targets)
        ticks,targets=cache[case];where=np.flatnonzero(ticks==int(r['control_tick']))
        if len(where)!=1 or where[0]==0 or ticks[where[0]]-ticks[where[0]-1]!=1:
            unmatched.append(dict(case=case,clock_s=r['clock_s'],reason='no adjacent command pair'));continue
        index=int(where[0]);previous=targets[index-1];final=targets[index]
        previous_error=float(np.max(np.abs(previous-np.asarray(r['previous_dispatched_target']))))
        final_error=float(np.max(np.abs(final-np.asarray(r['final_dispatched_joint_target_rad']))))
        arm_step=float(np.max(np.abs(final[12:]-previous[12:])))
        slew_violation=max(0.,arm_step-a.arm_target_rate*.01)
        rate_error=abs(float(r['arm_command_rate_rad_s'])-a.arm_target_rate)
        checks.append(dict(case=case,tick=int(ticks[index]),applied=bool(r.get('applied_first_step')),
            previous_command_error_rad=previous_error,final_command_error_rad=final_error,
            trace_arm_step_max_rad=arm_step,trace_arm_slew_violation_rad=slew_violation,
            declared_rate_error_rad_s=rate_error))
    maximum=lambda key:max((x[key] for x in checks),default=None)
    violated=[x for x in checks if max(x['previous_command_error_rad'],x['final_command_error_rad'],
                 x['trace_arm_slew_violation_rad'])>a.tolerance_rad or x['declared_rate_error_rad_s']>1e-9]
    for f in ('INPUTS.json','metrics.json','job.json','landing_mpc_records.json','native_command_inputs.json'):
        hash_file(a.run/f)
    passed=bool(checks) and not violated and not exceptions and not unmatched
    summary=dict(scope='Independent first-episode command-history mapping check, not a dynamics or recovery certificate',
        run=str(a.run),passed=passed,arm_target_rate_rad_s=a.arm_target_rate,tolerance_rad=a.tolerance_rad,
        complete5=metrics['complete5'],records=len(records),matched_records=len(checks),
        applied_records=sum(x['applied'] for x in checks),student_fallback_records=sum(not x['applied'] for x in checks),
        previous_command_error_max_rad=maximum('previous_command_error_rad'),
        final_command_error_max_rad=maximum('final_command_error_rad'),
        trace_arm_slew_violation_max_rad=maximum('trace_arm_slew_violation_rad'),
        exceptions=exceptions,unmatched=unmatched,violations=violated,source_sha256=hashes)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','violations','exceptions','unmatched')}))
    return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
