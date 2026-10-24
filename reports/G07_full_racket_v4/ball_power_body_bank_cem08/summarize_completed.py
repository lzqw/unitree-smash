"""Recompute the stopped native search's completed generations and resource use."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[2]))
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def count(rows):
    keys = ['had_contact', 'frame_hit', 'airborne_string_hit', 'recovery_success',
            'crossed_net', 'landed_legal_court', 'landed_target', 'attack_outgoing',
            'complete_smash', 'robot_net_fault', 'racket_invasion_fault']
    result = {key: sum(bool(row[key]) for row in rows) for key in keys}
    result.update(executions=len(rows),
                  air_strings_and_recovery=sum(row['airborne_string_hit'] and row['recovery_success'] for row in rows),
                  task_valid_air_strings_crosses=sum(usable_air_contact(row) and row['crossed_net'] for row in rows),
                  robot_rule_terminal=sum(row['training_terminal_reasons']['robot_rule_fault'] for row in rows),
                  body_ground_terminal=sum(row['training_terminal_reasons']['body_ground'] for row in rows),
                  numerical_failures=sum(row['numerical_failures'] for row in rows))
    return result


metrics = json.loads((RUN / 'metrics.json').read_text())
job = json.loads((RUN / 'job.json').read_text())
interrupted = json.loads((RUN / 'interrupted.json').read_text())
assert job['status'] == 'FAILED' and not job['receipt_present']
history, all_rows, source_files = [], [], ['metrics.json', 'job.json', 'interrupted.json']
for entry in metrics['history']:
    generation = entry['generation']
    name = f'episodes_{generation:03}.json'
    rows = json.loads((RUN / name).read_text())
    assert len(rows) == 128 and all(row['episode_index'] == 0 for row in rows)
    history.append(dict(count(rows), generation=generation,
                        control_ticks=entry['control_ticks'], env_steps=entry['env_steps']))
    all_rows += [dict(row, generation=generation) for row in rows]
    source_files.append(name)
guided = [row for row in all_rows if row['generation'] == 0 and 2 <= row['candidate_id'] <= 61]
actual_steps = sum(entry['env_steps'] for entry in history) + interrupted['control_ticks'] * 128
report = dict(created_at=datetime.now(timezone.utc).isoformat(),
              scope='Two completed native TRAIN generations plus an interrupted21tick next generation. Four-generation job intentionally stopped; not formal acceptance.',
              counts=count(all_rows), history=history, first_generation_guided_proposals=count(guided),
              whole_job_complete=False, requested_generations=4, completed_generations=len(history),
              guard_terminal_state=job['status'], actual_gpu_hours=job['charged_gpu_hours'],
              actual_control_env_steps=actual_steps, charged_control_env_step_cap=job['charged_env_steps'],
              interrupted=interrupted, source_commit=job['git_sha'], formal_acceptance=False,
              reason='No attack/full success or joined air-string recovery; forward body proposals predominantly invalidate the rally before impact through net/invasion faults.',
              next_method='Public task-space Cartesian preparation, with unchanged asset, native rules, actuator limits and success gates.',
              source_sha256={name: hashlib.sha256((RUN / name).read_bytes()).hexdigest() for name in source_files})
atomic_json(RUN / 'STOPPED_TRAIN_SUMMARY.json', report)
keys = ['generation', 'env_id', 'candidate_id', 'source_training_index', 'had_contact',
        'airborne_string_hit', 'frame_hit', 'crossed_net', 'landed_legal_court', 'landed_target',
        'attack_outgoing', 'recovery_success', 'complete_smash', 'robot_net_fault',
        'racket_invasion_fault', 'numerical_failures']
with (RUN / 'episodes.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: row[key] for key in keys} for row in all_rows)
print(json.dumps(report, ensure_ascii=False))
