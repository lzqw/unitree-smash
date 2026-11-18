"""Review the fixed model fit against unchanged references; never promote."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
fit=args.run/'fit';output=args.run/'review.json'
if output.exists():raise FileExistsError(output)
root=args.run.parent
completed=json.loads((fit/'completed.json').read_text())
config=json.loads((fit/'config.json').read_text())
before=json.loads((fit/'validation_initial.json').read_text())
after=json.loads((fit/'validation_final.json').read_text())
baseline_path=root/'native_response_fit_01/validation_initial.json'
original=json.loads(baseline_path.read_text())['legacy']
history=[json.loads(line) for line in (fit/'history.jsonl').read_text().splitlines()]
if [x['update'] for x in history]!=list(range(1,257)) or any(x['actor_updates']!=0 for x in history):
    raise ValueError('Incomplete or mixed model update history')
for row in history:
    if any(not math.isfinite(row[k]) for k in ('loss','gradient','foot_contact_gradient','ball_gate_gradient')):
        raise FloatingPointError('Nonfinite model history')
for key,name in [('world_sha256','world.pt'),('action_ball_sha256','action_ball_candidate.pt')]:
    if hashlib.sha256((fit/name).read_bytes()).hexdigest()!=completed[key]:raise ValueError('Changed output checkpoint '+name)
legacy_path=fit/'ball.pt'
if not legacy_path.exists():
    legacy_path=Path(completed['legacy_ball_path'])
if not legacy_path.exists():
    raise ValueError('Missing reused legacy ball checkpoint '+str(legacy_path))
if hashlib.sha256(legacy_path.read_bytes()).hexdigest()!=completed['legacy_ball_sha256']:
    raise ValueError('Changed output/reused legacy ball checkpoint '+str(legacy_path))
inventory=json.loads((args.run/'collected_response_inventory.json').read_text())
sources=[]
for item in inventory['sources']:
    name=item['path'];a=before['responses'][name];b=after['responses'][name]
    aa={r['lane']:r for r in a['rows']};bb={r['lane']:r for r in b['rows']}
    contacts=item['groups']['validation_lanes']['nominal_contact_lanes']
    rows=[dict(lane=lane,before=aa[lane]['errors']['ball_velocity'],after=bb[lane]['errors']['ball_velocity']) for lane in contacts]
    angular=[]
    if item['head']=='recovery':
        angular=[dict(lane=lane,before=aa[lane]['errors']['root_angular_velocity'],after=bb[lane]['errors']['root_angular_velocity']) for lane in sorted(bb)]
    sources.append(dict(path=name,head=item['head'],tick=item['tick'],heldout_contact_response=rows,
        recovery_angular_response=angular,foot_before=a.get('foot_contact'),foot_after=b.get('foot_contact'),
        saturated_commands=b['saturated_count'],all_heldout_responses=b['rows']))
contact=[r for s in sources for r in s['heldout_contact_response']]
if any(after['candidate'][key]['count']!=original[key]['count'] for key in ('-1','0','1')):
    raise ValueError('Event-aligned comparison cohort changed')
old_sign_errors=round(original['0']['forward_sign_error']*original['0']['count'])
new_sign_errors=round(after['candidate']['0']['forward_sign_error']*after['candidate']['0']['count'])
checks=dict(contact_velocity_improved=after['candidate']['0']['velocity_rmse']<original['0']['velocity_rmse'],
    contact_sign_error_improved=new_sign_errors<old_sign_errors,
    precontact_not_worse=after['candidate']['-1']['velocity_rmse']<=original['-1']['velocity_rmse'],
    postcontact_not_worse=after['candidate']['1']['velocity_rmse']<=original['1']['velocity_rmse'])
summary=dict(original_legacy_comparison=checks,
    heldout_contact_pairs=len(contact),
    positive_contact_response_directions=sum(r['after']['cosine'] is not None and r['after']['cosine']>0 for r in contact),
    negative_contact_response_directions=sum(r['after']['cosine'] is not None and r['after']['cosine']<0 for r in contact),
    undefined_contact_response_directions=sum(r['after']['cosine'] is None for r in contact),
    actual_model_updates=len(history),actor_updates=0,critic_updates=0,
    median_update_seconds=statistics.median(r['seconds'] for r in history),elapsed_seconds=completed['elapsed_s'],
    no_automatic_promotion=True,native_admission=False)
result=dict(status='PREDICTION_GATE_FAILED' if not all(checks.values()) else 'PREDICTION_GATE_PASSED_FURTHER_ADMISSION_REQUIRED',
    summary=summary,completion=completed,original_legacy=original,
    original_legacy_file=str(baseline_path),original_legacy_file_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    run_initial_candidate=before['candidate'],current=after['candidate'],
    candidate_schema=config.get('action_ball_schema','action-conditioned-racket-sweep-v1'),
    initialization_migrated_schema=config.get('action_ball_migration',False),
    run_local_reference=before['legacy'],sources=sources,autoregressive=after['autoregressive'],
    limitations=['Model step count is not actor updates or native success.',
        'Direction counts are diagnostics, not an independently calibrated validity region.',
        'Single direction and sparse measured times do not prove complete head/contact coverage.',
        'Task memory feedback, current critic trust, real R-A-R and training resume remain unverified.'])
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],**summary,current=after['candidate']),indent=2))
