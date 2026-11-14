#!/usr/bin/env python3
"""Audit native return and event ordering for the accepted and bad policies.

This is a file-only audit. It never treats a model return as a native return
and never changes the task score. Missing reward-component traces are reported
explicitly rather than reconstructed from binary task flags.
"""
import argparse, hashlib, json, statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(directory):
    directory=Path(directory)
    rows=json.loads((directory/'episodes_summary.json').read_text())
    metrics=json.loads((directory/'metrics.json').read_text())
    done=json.loads((directory/'completed.json').read_text())
    if len(rows)!=128 or done.get('control_ticks')!=800:
        raise ValueError(f'incomplete native panel: {directory}')
    rows=sorted(rows,key=lambda x:int(x['source_case_index']))
    if [int(x['source_case_index']) for x in rows]!=list(range(128)):
        raise ValueError('case ordering is not the fixed original128 order')
    return metrics,rows


def counts(rows):
    return {key:sum(bool(row.get(key)) for row in rows) for key in (
        'qualified_airborne_string_contact','ball_side_complete_except_recovery',
        'recovery_success','complete5','body_ground_seen','rally_fault')}


def set_delta(a,b,key):
    x={i for i,r in enumerate(a) if r.get(key)};y={i for i,r in enumerate(b) if r.get(key)}
    return {'initial':len(x),'candidate':len(y),'retained':sorted(x&y),
            'lost':sorted(x-y),'new':sorted(y-x)}


def rejection_delta(a,b):
    keys=sorted(set(k for r in a+b for k in r.get('rejection_reasons',[])))
    return {key:{'initial':sum(key in r.get('rejection_reasons',[]) for r in a),
                 'candidate':sum(key in r.get('rejection_reasons',[]) for r in b)} for key in keys}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-eval',type=Path,required=True)
    p.add_argument('--bad-eval',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--bad-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rm,a=load(args.reference_eval);bm,b=load(args.bad_eval)
    if any(x['case_id']!=y['case_id'] for x,y in zip(a,b)):raise ValueError('case IDs differ')
    returns=[float(y['native_episode_return'])-float(x['native_episode_return']) for x,y in zip(a,b)]
    component_traces=all('reward_components' in row for row in a+b)
    report=dict(
        reference=dict(path=str(args.reference_checkpoint.resolve()),sha256=sha(args.reference_checkpoint),metrics=rm,counts=counts(a)),
        bad=dict(path=str(args.bad_checkpoint.resolve()),sha256=sha(args.bad_checkpoint),metrics=bm,counts=counts(b)),
        native_return=dict(reference_mean=statistics.mean(r['native_episode_return'] for r in a),
            bad_mean=statistics.mean(r['native_episode_return'] for r in b),
            mean_delta=statistics.mean(returns),median_delta=statistics.median(returns),
            positive_cases=sum(x>0 for x in returns),negative_cases=sum(x<0 for x in returns)),
        per_case={key:set_delta(a,b,key) for key in (
            'qualified_airborne_string_contact','ball_side_complete_except_recovery',
            'recovery_success','complete5')},
        rejection_reason_counts=rejection_delta(a,b),
        reward_component_decomposition=dict(available=component_traces,
            status='native reward component traces present' if component_traces else
                   'unavailable: episodes_summary has terminal task evidence but no per-step reward components; no numeric decomposition was fabricated'),
        audit_contract=dict(native_return='undiscounted sum of native control-step rewards',
            imagined_return_comparison='not performed; horizon/bootstrap do not match 8-second native return',
            model_event_accuracy='requires saved event prediction traces; terminal summaries alone are insufficient',
            critic_calibration='requires native continuation values aligned to the same state and policy'))
    (args.output/'calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['native_return'],ensure_ascii=False))


if __name__=='__main__':main()
