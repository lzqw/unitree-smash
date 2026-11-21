"""Relate a measured native arrival objective to frozen formal outcomes on the same initial cases."""
import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.evaluate_supermimic_fresh_wm_panel import digest


def outcome_table(cases, outcome):
    positive=[r['terms'] for r in cases if r[outcome]]
    negative=[r['terms'] for r in cases if not r[outcome]]
    summary=dict(positive=len(positive), negative=len(negative))
    if not positive or not negative:
        return summary
    for i,name in enumerate(('position','linear_velocity','normal','angular_velocity','sum')):
        def loss(x):
            return sum(x) if name=='sum' else x[i]
        good=list(map(loss,positive)); bad=list(map(loss,negative))
        rank=sum((a<b)+0.5*(a==b) for a in good for b in bad)/(len(good)*len(bad))
        summary[name]=dict(positive_median=statistics.median(good),negative_median=statistics.median(bad),
                           lower_loss_ranks_positive_auc=rank)
    return summary


def audit(root, costs, evaluation):
    read=lambda f:json.loads(f.read_text())
    result=read(costs)
    native=root/'raw'
    expected=result['native_panels']['disjoint']
    if (not result['native_arrival_substeps'] or digest(native/'status.json')!=expected
            or read(native/'status.json')['status']!='COLLECTION_FINISHED'):
        raise ValueError('Substep readout and completed collection differ')
    metadata=read(evaluation/'metrics.json')
    if (not metadata['actor_parameters_unchanged'] or metadata['actor_state_sha256']!=result['actor_sha256']
            or metadata['control_ticks']!=800 or metadata['num_envs']!=128):
        raise ValueError('Require the same frozen policy and complete native formal protocol')
    spec_dir,=native.glob('spec_*')
    for name in ('initial_states.json','task_contract.json','policy_execution.json','physics.json'):
        if digest(spec_dir/'nominal_a'/name)!=digest(evaluation/name):
            raise ValueError('Native and formal initial conditions or execution differ')
    spec=read(spec_dir/'nominal_a/native_probe_manifest.json')['spec']
    receipts=list(map(json.loads,(evaluation/'episodes.jsonl').read_text().splitlines()))
    final={r['env_id']:r for r in receipts}
    if len(receipts)!=128 or set(final)!=set(range(128)):
        raise ValueError('Expected 128 distinct original episodes')
    scores={}
    for name, entry in result['world_then_reference']['after'].items():
        cases=[]
        for row in entry['disjoint']['cases']:
            rec=final[row['lane']]
            contact=rec['contact_time_s']
            if rec['had_contact'] and (contact is None or not math.isfinite(contact)):
                raise ValueError('Latched contact has no finite native timestamp')
            cases.append(dict(lane=row['lane'], terms=row['nominal_native_terms'], tick=row['tick'], native_arrival_time_s=row['arrival_time_s'],
                      precontact_at_interval_start=row['native_precontact_by_branch']['nominal_a'],
                      all_native_branches_precontact=all(row['native_precontact_by_branch'].values()),
                      native_first_contact_s=contact if rec['had_contact'] else None, frame_hit=rec['frame_hit'],
                      qualified=rec['qualified_airborne_string_contact'], ball_side=rec['ball_side_complete_except_recovery'],
                      correct_direction=row['model_derivative']*row['measured']>0,supported=row['supported']))
        if len({r['lane'] for r in cases})!=len(cases) or not {r['lane'] for r in cases}<=set(spec['lanes']):
            raise ValueError('Arrival case identity mismatch')
        groups={'all_measured':cases,'precontact_at_interval_start':[r for r in cases if r['precontact_at_interval_start']],
                'all_branches_precontact':[r for r in cases if r['all_native_branches_precontact']]}
        scores[name]=dict(registered=len(spec['lanes']),missing=sorted(set(spec['lanes'])-{r['lane'] for r in cases}),
                          cases=cases,groups={group:{key:outcome_table(rows,key) for key in ('qualified','ball_side')}
                          for group,rows in groups.items()})
    return dict(scores=scores,raw_status_sha256=expected,formal_episodes_sha256=digest(evaluation/'episodes.jsonl'),
                costs_sha256=digest(costs),scope='Observed association of absolute costs from the frozen nominal branch with the fixed-case outcome. No intervention on the costs or policy update.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','costs','evaluation','output'):
        cli.add_argument('--'+key,required=True,type=Path)
    args=cli.parse_args()
    result=audit(args.collection,args.costs,args.evaluation)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
