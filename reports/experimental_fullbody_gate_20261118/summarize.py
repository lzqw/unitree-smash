"""Summarize fixed replay evidence; no controller or native execution."""
import gzip,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
p=root/'REPLAY.json'
x=json.loads(p.read_text()) if p.exists() else json.load(gzip.open(str(p)+'.gz','rt'))
rows=[];timings=[]
for plan in x['plans']:
    clocks=dict(zip(plan['env_ids'],plan['observation_clock_s']))
    for s in plan['contact_first_plan_summaries']:
        timings.append(s['planning_seconds'])
        for a in s.get('search_diagnostics',{}).get('anchors',[]):
            trial=a.get('task_joint_search',{})
            chosen=trial.get('selected') or trial.get('best_failed') or {}
            row=dict(lane=s['lane'],clock_s=clocks[s['lane']],duration_s=a['contact_time_s'],
                anchor_error_m=a['position_error_anchor_m'],
                ATTACK_GEOMETRY_FEASIBLE=a.get('ATTACK_GEOMETRY_FEASIBLE',False),
                initial_ANALYTIC_GOAL_VALID=a.get('initial_ANALYTIC_GOAL_VALID',False),
                joint_contact_accepted=trial.get('accepted',False),
                rejection=a.get('rejection'),
                stages={k:chosen.get(k,False) for k in ('POSITION_FEASIBLE','NORMAL_FEASIBLE','ANALYTIC_GOAL_VALID','TASK_FEASIBLE_RACKET_VELOCITY')},
                physical_result=a.get('physical_result'))
            rows.append(row)
r=x['physical_calls'][0]['result']
summary=dict(fixed_events=len(rows),planning_calls=len(timings),
    analytic_joint_contacts=sum(y['joint_contact_accepted'] for y in rows),
    fullbody_calls=len(x['physical_calls']),
    fullbody_prediction_valid=sum(c['result']['accepted'] for c in x['physical_calls']),
    path_intervals_completed=r.get('prediction_intervals'),terminal_error_m=r.get('actual_racket_position_error_m'),
    terminal_normal_error=r.get('actual_racket_normal_error'),terminal_velocity_error_m_s=r.get('actual_racket_velocity_error_m_s'),
    rejection=r['rejection'],planning_seconds=timings,qualified_bundles=len(x['offline_reference_bundles']),
    PHYSX_REFERENCE_VALID='NOT_RUN',NATIVE_HIT='NOT_RUN',RECOVERY_SUCCESS='NOT_RUN',COMPLETE5='NOT_RUN',
    native_execution=False,formal_cases_read=False,training=False)
(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'STAGES.json').write_text(json.dumps(rows,indent=2)+'\n')
manifest=json.loads((root/'MANIFEST.json').read_text())
checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in manifest['protected_sha256'].items()}
source={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in manifest['source_sha256'].items() if not p.startswith('tests/')}
assert all(checks.values()) and all(source.values())
(root/'FREEZE_CHECK.json').write_text(json.dumps(dict(protected=checks,control_source=source,tests_added_after_replay=True),indent=2)+'\n')
print(json.dumps(summary,indent=2))
