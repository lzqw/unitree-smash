"""Recompute the stopped measured-velocity search's completed generations and resource use."""
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
    result.update(executions=len(rows), usable_air_strings=sum(usable_air_contact(row) for row in rows),
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
              scope='Completed measured-velocity generation plus an interrupted next generation. Four-generation job intentionally stopped; not formal acceptance.',
              counts=count(all_rows), history=history, first_generation_guided_proposals=count(guided),
              whole_job_complete=False, requested_generations=4, completed_generations=len(history),
              guard_terminal_state=job['status'], actual_gpu_hours=job['charged_gpu_hours'],
              actual_control_env_steps=actual_steps, charged_control_env_step_cap=job['charged_env_steps'],
              interrupted=interrupted, source_commit=job['git_sha'], formal_acceptance=False,
              reason='Generation0had7contacts/1airstrings/0qualified recoveries/0attacks/full/NF0;114body-ground terminations and14net faults. Guided measured-rate directions produce forward-moving physical frame contacts but lose strings alignment and body stability.',
              next_method='Retain CEM14contact9/recovery7, blend original rates with measured-forward CEM15frame candidate35, move interception to own-side1.05/1.15/1.25m, reduce commanded point velocity to1.5m/s, initialize event calibration0and independently search arm lead0/.04/.08s. The actual outgoing ball gate stays8m/s; all physics/hardware/rules/gates unchanged.',
              source_sha256={name: hashlib.sha256((RUN / name).read_bytes()).hexdigest() for name in source_files})
partial_rallies = json.loads((RUN / interrupted['terminal_rally_records']).read_text())
partial_jumps = {r['env_id']: r for r in json.loads((RUN / interrupted['terminal_jump_records']).read_text())}
partial_rows = []
for row in partial_rallies:
    if row['episode_index'] != 0:
        continue
    velocity = row['outgoing_velocity_mps']
    attack = bool(row['released'] and row['legal_initial_contact'] and row['string_hit'] and row['airborne_string_hit']
                  and all(__import__('math').isfinite(v) for v in velocity)
                  and sum(v*v for v in velocity) >= 64 and velocity[0] > 0 and velocity[2] <= -1)
    partial_rows.append(dict(row, jump=partial_jumps[row['env_id']], attack_outgoing=attack))
assert len({r['env_id'] for r in partial_rows}) == len(partial_rows)
report['interrupted_closed_first_episodes'] = count(partial_rows)
report['interrupted_pending_first_episodes'] = 128-len(partial_rows)
report['interrupted_observed_contact_events'] = len(json.loads((RUN / f"contacts_{interrupted['generation']:03d}.json").read_text()))
report['interrupted_scope'] = 'Closed first episodes only; pending episodes not treated as success or completed failure. Interrupted population/terminal records/poses preserve actual398ticks, distinct from periodic300tick snapshot.'
for name in [interrupted['terminal_rally_records'], interrupted['terminal_jump_records'], interrupted['actual_partial_population']]:
    report['source_sha256'][name] = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
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
