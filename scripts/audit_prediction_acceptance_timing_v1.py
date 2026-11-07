#!/usr/bin/env python3
"""Locate model acceptance relative to actual first nonfoot force events.

An accepted prediction after a historical task fault cannot rescue that rally.
An event within a previously accepted horizon is also not an open-loop model
counterexample: native receding-horizon execution replans subsequent commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory


def counts(records,event_s):
    accepted=[r for r in records if r['accepted']]
    before=[r for r in accepted if .01*r['control_tick']<event_s-1e-9]
    after=[r for r in accepted if .01*r['control_tick']>=event_s-1e-9]
    overlap=[r for r in before if event_s<=.01*(r['control_tick']+r['horizon_steps'])+1e-9]
    return dict(records=len(records),applied=sum(bool(r.get('applied_first_step')) for r in records),
        accepted=len(accepted),accepted_before_nonfoot=len(before),accepted_after_nonfoot=len(after),
        accepted_before_event_with_event_in_horizon=len(overlap),
        applied_before_nonfoot=int(sum(bool(r.get('applied_first_step')) and .01*r['control_tick']<event_s-1e-9 for r in records)),
        first_applied_s=next((.01*r['control_tick'] for r in records if r.get('applied_first_step')),None),
        first_accepted_s=next((.01*r['control_tick'] for r in accepted),None),
        first_nonfoot_s=float(event_s) if np.isfinite(event_s) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    read=lambda n:json.loads((a.run/n).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed native run required')
    history=RecordedNonfootContactHistory(a.run);records=read('landing_mpc_records.json')
    outcomes={r['training_index']:r for r in read('metrics.json')['complete5_case_outcomes']}
    if any(r['episode_index']!=0 or abs(r['clock_s']-.01*r['control_tick'])>2e-6 for r in records):
        raise ValueError('Require matching first-episode command clocks')
    rows=[]
    for case in range(128):
        subset=[r for r in records if r['case']==case]
        rows.append(dict(case=case,complete5=outcomes[case]['complete5'],
            ball5_except_recovery=outcomes[case]['ball5_except_recovery'],
            **counts(subset,history.first_frame[case]*history.dt)))
    totals={key:sum(r[key] for r in rows) for key in ['records','applied','accepted',
        'accepted_before_nonfoot','accepted_after_nonfoot',
        'accepted_before_event_with_event_in_horizon','applied_before_nonfoot']}
    files=[Path(__file__),ROOT/'scripts/recorded_nonfoot_contact_history_v1.py']+[a.run/n for n in
        ['job.json','metrics.json','landing_mpc_records.json','contact_telemetry_metadata.json',
         'first_nonfoot_contact_details.json','fixed_policy_traces/manifest.json']]
    result=dict(scope=__doc__,run=str(a.run.resolve()),totals=totals,case_results=rows,
        nonfoot_threshold_n=1.,event_physics_dt_s=history.dt,
        limitations=['Nonfoot event only, not all other historical rule faults.',
            'Same rollout correlation, not teacher/student counterfactual causality.',
            'No whole planned command-sequence replay in the native model.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(totals))


if __name__=='__main__':main()
