#!/usr/bin/env python3
"""Separate native recovery outcomes from prediction and solver diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run


def distribution(values):
    values=np.asarray([float(x) for x in values if x is not None],dtype=float)
    if not len(values):return dict(count=0,mean=None,median=None,p95=None,maximum=None)
    if not np.isfinite(values).all():raise ValueError('Nonfinite diagnostic')
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),
        p95=float(np.quantile(values,.95)),maximum=float(values.max()))


def summarize_records(records):
    first_reject=[];later_reject=[];terminal_reject=[];other_reject=[]
    for row in records:
        if row.get('accepted'):continue
        stage=row.get('failed_stage')
        if stage is not None:
            (first_reject if stage==0 else later_reject).append(row)
        elif row.get('first_action_available'):
            terminal_reject.append(row)
        else:other_reject.append(row)
    return dict(records=len(records),applied=sum(bool(r.get('applied_first_step')) for r in records),
        first_actions_available=sum(bool(r.get('first_action_available')) for r in records),
        full_horizon_accepted=sum(bool(r.get('accepted')) for r in records),
        first_stage_rejected=len(first_reject),later_stage_rejected=len(later_reject),
        post_rollout_rejected=len(terminal_reject),other_rejected=len(other_reject),
        controller_rejection_reasons=dict(Counter(str(r.get('rejection')) for r in records if not r.get('accepted'))),
        failed_stage_histogram=dict(Counter(str(r.get('failed_stage')) for r in records if r.get('failed_stage') is not None)),
        solver_status_histogram=dict(Counter(str(r.get('failed_solver_status')) for r in records if r.get('failed_stage') is not None)),
        # Evaluator only records the actual next control tick, never held-target ages.
        next_tick_joint_position_prediction_error_rad=distribution(
            r.get('prediction_joint_position_error_max_rad') for r in records),
        command_tracking_error_rad=distribution(r.get('command_tracking_error_max_rad') for r in records),
        solver_wall_ms=distribution(r.get('runtime_ms') for r in records),
        applied_nonimproving_risk=sum(bool(r.get('applied_first_step')) and
            r.get('first_step_risk_score_drop') is not None and r['first_step_risk_score_drop']<=0 for r in records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--command-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous result')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    outcomes,_=read_run(args.run,protocol)
    audit=json.loads(args.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=args.run.resolve():
        raise ValueError('Matching native command audit required')
    # Bind the supplied audit to the exact files it checked, not just its passed flag.
    for path,expected in audit['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Audited native evidence changed: '+path)
    records=json.loads((args.run/'landing_mpc_records.json').read_text())
    if any(r.get('episode_index')!=0 for r in records):raise ValueError('Not first episodes')
    episodes={r['source_training_index']:r for r in json.loads((args.run/'episodes.json').read_text())}
    stages={r['training_index']:r for r in outcomes['stages']}
    selected=json.loads((args.run/'INPUTS.json').read_text())['landing_mpc_cases']
    rows=[]
    for case in selected:
        row=dict(case=case,**stages[case],**summarize_records([r for r in records if r['case']==case]))
        physical=episodes[case]['contact_flight_recovery']
        row.update(stable_recovery_s=physical['recovery_s'],body_landing=physical['body_landing'],
            terminal_root_height_m=physical['terminal_root_height_m'])
        rows.append(row)
    paths=[Path(__file__),args.command_audit,protocol_path]+[args.run/x for x in
        ('INPUTS.json','metrics.json','episodes.json','landing_mpc_records.json','job.json','completed.json')]
    result=dict(schema='native-landing-execution-breakdown-v1',run=str(args.run.resolve()),
        complete5=outcomes['complete5'],air=outcomes['air'],recovery=outcomes['recovery'],
        ball5_except_recovery=outcomes['ball5_except_recovery'],numerical_failures=outcomes['numerical_failures'],
        summary=summarize_records(records),cases=rows,
        scope='Recorded native feedback execution. Prediction errors occur on each controller own states; not a paired model counterfactual. Wall time is not injected physical latency.',
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cases')}))


if __name__=='__main__':main()
