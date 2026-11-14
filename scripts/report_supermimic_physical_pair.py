#!/usr/bin/env python3
"""Paired raw native returns; no reruns or score-definition changes."""
import argparse,json,hashlib,csv,statistics
from pathlib import Path


def load(directory):
    metrics=json.loads((directory/'metrics.json').read_text())
    done=json.loads((directory/'completed.json').read_text())
    rows=json.loads((directory/'episodes_summary.json').read_text())
    if done['control_ticks']!=800 or done['episodes']!=128 or len(rows)!=128:
        raise ValueError('Incomplete original128 evaluation')
    if sorted(int(r['source_case_index']) for r in rows)!=list(range(128)):
        raise ValueError('Changed/missing case mapping')
    rows=sorted(rows,key=lambda r:r['source_case_index'])
    if any(r['native_episode_return'] is None for r in rows):raise ValueError('Missing raw native return')
    return metrics,rows


def compare_set(a,b,key):
    x={i for i,r in enumerate(a) if r[key]};y={i for i,r in enumerate(b) if r[key]}
    return dict(initial=len(x),candidate=len(y),retained=sorted(x&y),lost=sorted(x-y),new=sorted(y-x))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('initial','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    am,a=load(args.initial);bm,b=load(args.candidate)
    if am['task_contract']!=bm['task_contract']:raise ValueError('Task contract mismatch')
    if any(x['case_id']!=y['case_id'] for x,y in zip(a,b)):raise ValueError('Case IDs differ')
    rows=[]
    for i,(x,y) in enumerate(zip(a,b)):
        rows.append(dict(case_index=i,case_id=x['case_id'],initial_return=x['native_episode_return'],
            candidate_return=y['native_episode_return'],return_delta=y['native_episode_return']-x['native_episode_return'],
            initial_contact=x['qualified_airborne_string_contact'],candidate_contact=y['qualified_airborne_string_contact'],
            initial_ball=x['ball_side_complete_except_recovery'],candidate_ball=y['ball_side_complete_except_recovery'],
            initial_recovery=x['recovery_success'],candidate_recovery=y['recovery_success'],
            initial_complete5=x['complete5'],candidate_complete5=y['complete5'],
            initial_head_contact=x.get('head_contact'),candidate_head_contact=y.get('head_contact'),
            initial_nonfoot_contact=x.get('independent_nonfoot_contact'),candidate_nonfoot_contact=y.get('independent_nonfoot_contact')))
    delta=[r['return_delta'] for r in rows]
    result=dict(initial_return_mean=statistics.mean(r['initial_return'] for r in rows),
        candidate_return_mean=statistics.mean(r['candidate_return'] for r in rows),mean_paired_delta=statistics.mean(delta),
        median_paired_delta=statistics.median(delta),positive_cases=sum(v>0 for v in delta),negative_cases=sum(v<0 for v in delta),
        task_sets={key:compare_set(a,b,key) for key in ('qualified_airborne_string_contact','ball_side_complete_except_recovery','recovery_success','complete5')},
        initial_metrics=am,candidate_metrics=bm,
        note='Frozen native original128 development panel. No statistical/generalization claim from one seed.')
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'paired_returns.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('task_sets','initial_metrics','candidate_metrics')}))


if __name__=='__main__':main()
