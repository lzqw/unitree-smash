"""Recompute actual completed CEM19generations and saved-stop resource evidence."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parents[2]))
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact

def count(rows):
    keys=['had_contact','frame_hit','airborne_string_hit','recovery_success','crossed_net',
          'landed_legal_court','landed_target','attack_outgoing','complete_smash',
          'robot_net_fault','racket_invasion_fault']
    result={k:sum(bool(row[k]) for row in rows) for k in keys}
    result.update(executions=len(rows),usable_air_strings=sum(usable_air_contact(r) for r in rows),
        air_strings_and_recovery=sum(r['airborne_string_hit'] and r['recovery_success'] for r in rows),
        attack_outgoing_and_recovery=sum(r['attack_outgoing'] and r['recovery_success'] for r in rows),
        task_valid_air_strings_crosses=sum(usable_air_contact(r) and r['crossed_net'] for r in rows),
        robot_rule_terminal=sum(r['training_terminal_reasons']['robot_rule_fault'] for r in rows),
        body_ground_terminal=sum(r['training_terminal_reasons']['body_ground'] for r in rows),
        numerical_failures=sum(r['numerical_failures'] for r in rows))
    return result

metrics=json.loads((RUN/'metrics.json').read_text())
job=json.loads((RUN/'job.json').read_text())
interrupted=json.loads((RUN/'interrupted.json').read_text())
assert job['status']=='FAILED' and not job['receipt_present'] and not job['timed_out']
source_files=['metrics.json','job.json','interrupted.json']
all_rows,history=[],[]
for entry in metrics['history']:
    name=f"episodes_{entry['generation']:03d}.json"
    rows=json.loads((RUN/name).read_text())
    assert len(rows)==entry['episodes']==128 and all(r['episode_index']==0 for r in rows)
    all_rows.extend(dict(r,generation=entry['generation']) for r in rows)
    history.append(dict(count(rows),generation=entry['generation'],control_ticks=entry['control_ticks'],env_steps=entry['env_steps']))
    source_files.append(name)
actual_steps=sum(h['env_steps'] for h in history)+interrupted['control_ticks']*128
report=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual completed early-landing37generation(s), followed by saved interruption. Not whole four-generation completion or formal acceptance.',
    source_commit=job['git_sha'],counts=count(all_rows),history=history,
    first_generation_guided_proposals=count([r for r in all_rows if r['generation']==0 and 2<=r['candidate_id']<=61]),
    whole_job_complete=False,requested_generations=4,completed_generations=len(history),
    guard_terminal_state=job['status'],actual_gpu_hours=job['charged_gpu_hours'],
    actual_control_env_steps=actual_steps,charged_control_env_step_cap=job['charged_env_steps'],interrupted=interrupted,
    reason='Earlier standing handover/extended leg targets retain actual attacking contacts but all airborne strings still fail recovery. Actual post-stroke net contact persists despite early foot support; target/full still0. Next hypothesis is earlier stronger arm retreat, not changed native rules.',
    next_method='Keep same61proposal numbers/reference asCEM19, tighten planned post-stroke full-tool command bound to1.15m and prediction to.12s after public forecast swing end. Same37search with38normal feature disabled.118CPUtests/122lane287tick preflight/four native-state61tick command replay; physical performance must be remeasured.',
    native_execution=True,formal_acceptance=False,
    source_sha256={n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in source_files})
path=RUN/'STOPPED_TRAIN_SUMMARY.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
keys=['generation','env_id','candidate_id','source_training_index','had_contact','airborne_string_hit',
      'frame_hit','crossed_net','landed_legal_court','landed_target','attack_outgoing','recovery_success',
      'complete_smash','robot_net_fault','racket_invasion_fault','numerical_failures']
with (RUN/'episodes.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
    writer.writerows({k:r[k] for k in keys} for r in all_rows)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','history']}))
