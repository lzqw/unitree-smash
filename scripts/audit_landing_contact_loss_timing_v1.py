#!/usr/bin/env python3
"""Compare first nonfoot contact with controller support rejection timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.summarize_landing_controller_execution_v1 import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous timing audit')
    read=lambda name:json.loads((a.run/name).read_text())
    job=read('job.json');inputs=read('INPUTS.json');done=read('completed.json')
    if job['status']!='SUCCEEDED' or done['env_steps']!=102400 or inputs['training_indices']!=list(range(128)):
        raise ValueError('Require completed original ordered TRAIN128')
    metadata=read('contact_telemetry_metadata.json');summary=read('contact_telemetry_summary.json')
    dt=metadata['physics_dt_s']
    # The native first ground reset starts the common physics counter at zero.
    # Match its endpoint to 800 actual 10ms commands before interpreting time.
    if abs(summary['last_physics_frame']*dt-8.)>1e-8:
        raise ValueError('Physical frame clock origin/endpoint not established')
    events=read('first_nonfoot_contact_details.json');records=read('landing_mpc_records.json')
    bodies={lane:event['physics_frame']*dt for event in events for lane in event['first_event_env_ids']}
    rows=[]
    for case in inputs['landing_mpc_cases']:
        case_records=[r for r in records if r['case']==case and r['episode_index']==0]
        rejected=[r['clock_s'] for r in case_records if r.get('rejection')=='insufficient support contacts for fixed-contact QP']
        body=bodies.get(case);first=min(rejected) if rejected else None
        # 2ms exclusion avoids boundary ambiguity between the 1ms physical
        # observer and the accumulated float32 control clock.
        pre=[r.get('prediction_joint_position_error_max_rad') for r in case_records
             if body is not None and r['clock_s']<body-.002]
        rows.append(dict(case=case,first_nonfoot_physical_time_s=body,
            first_no_support_controller_time_s=first,
            no_support_minus_nonfoot_s=first-body if first is not None and body is not None else None,
            prediction_error_before_nonfoot_rad=distribution(pre)))
    timed=[r for r in rows if r['no_support_minus_nonfoot_s'] is not None]
    result=dict(schema='native-contact-loss-timing-audit-v1',run=str(a.run.resolve()),
        cases=rows,body_and_support_rejection_cases=len(timed),
        support_rejection_after_body_by_over_2ms=sum(r['no_support_minus_nonfoot_s']>.002 for r in timed),
        support_rejection_before_body_by_over_2ms=sum(r['no_support_minus_nonfoot_s']<-.002 for r in timed),
        near_simultaneous_cases=sum(abs(r['no_support_minus_nonfoot_s'])<=.002 for r in timed),
        scope='Temporal ordering only; a later support rejection does not explain the earlier physical collapse. Prediction errors are recorded on each controller own states, not paired counterfactuals.',
        source_sha256={str((a.run/name).resolve()):hashlib.sha256((a.run/name).read_bytes()).hexdigest() for name in
            ('job.json','INPUTS.json','completed.json','contact_telemetry_metadata.json','contact_telemetry_summary.json',
             'first_nonfoot_contact_details.json','landing_mpc_records.json')})
    result['source_sha256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cases')}))


if __name__=='__main__':main()
