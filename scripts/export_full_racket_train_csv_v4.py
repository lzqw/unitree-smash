"""Export completed native TRAIN learning/assessment episodes with unchanged canonical outcomes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_prefix.resolve().with_suffix('.csv')
    manifest_path = output.with_suffix('.json')
    if output.exists() or manifest_path.exists():
        raise ValueError('Use a new export prefix; never overwrite prior evidence')
    job = json.loads((run / 'job.json').read_text())
    receipt = json.loads((run / 'completed.json').read_text())
    inputs = json.loads((run / 'INPUTS.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    learning = 'counts' in metrics and 'history' in metrics and 'runner' in inputs
    actual_steps = metrics['history'][-1]['actual_control_env_steps'] if learning else metrics['actual_control_env_steps']
    if job['status'] != 'SUCCEEDED' or receipt['env_steps'] != actual_steps:
        raise ValueError('Require an actually completed native TRAIN assessment')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol_sha = inputs['contract']['protocol_sha256'] if learning else inputs['protocol_sha256']
    policy_sha = sha(run / 'policy.pt') if learning else inputs['policy_sha256']
    if sha(protocol_path) != protocol_sha or policy_sha != metrics['policy_sha256']:
        raise ValueError('Runtime protocol/policy provenance mismatch')
    source_hashes = json.loads((run / 'source_hashes.json').read_text())
    for name in ['quadsmash/evaluation_v4.py', 'quadsmash/control/full_racket_train_evaluation_v4.py',
                 'quadsmash/physics/rally_rules_v4.py']:
        if sha(ROOT / name) != source_hashes[name]:
            raise ValueError('Canonical metric implementation changed; use original source snapshot')
    protocol = json.loads(protocol_path.read_text())
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol, inputs['training_indices'])
    rows_path = run / 'episodes.json'
    rows = json.loads(rows_path.read_text())
    if learning:
        if not rows or len({(r['env_id'], r['episode_index']) for r in rows}) != len(rows):
            raise ValueError('Each closed native training episode must appear exactly once')
        if any(not 0 <= r['env_id'] < inputs['num_envs'] or not r['ground_init'] or
               r['source_training_index'] != indices[r['env_id'] % len(indices)] for r in rows):
            raise ValueError('Original training case/ground-reset mapping differs')
        checked = [independent_episode(row, protocol) for row in rows]
        counts = dict(completed_episodes=len(rows), air=sum(r['airborne_contact'] for r in checked),
            conditional_recovery_count=sum(r['conditional_recovery'] for r in checked),
            complete=sum(r['complete'] for r in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
        if counts != metrics['counts']:
            raise ValueError('Changing-policy episode recomputation disagrees with the native run')
        summary = dict(episodes=len(rows), airborne_contacts=counts['air'],
            conditional_recovery_count=counts['conditional_recovery_count'], complete=counts['complete'],
            numerical_failures=counts['numerical_failures'], scope='Changing-policy training counts only; no frozen performance')
    else:
        summary = summarize_training(rows, cases, indices, protocol)
        if any(metrics[key] != value for key, value in summary.items()):
            raise ValueError('Independent canonical recomputation disagrees with runtime metrics')
    common = ['case_id', 'source_training_index', 'env_id', 'episode_index', 'ground_init',
        'active_rally_launch_count', 'numerical_failures', 'had_contact', 'legal_initial_contact',
        'string_hit', 'frame_hit', 'airborne_string_hit', 'released', 'crossed_net',
        'landed_legal_court', 'landed_target', 'contact_count', 'contact_time_s',
        'release_time_s', 'crossing_time_s', 'first_landing_time_s', 'net_margin_m']
    jump = ['total_com_rise_m', 'airborne_s', 'takeoff_time_s', 'landing_time_s', 'recovery_s',
        'qualified_jump_landed', 'stable_recovery', 'body_landing', 'terminated',
        'maximum_body_ground_force_n', 'max_leg_torque_nm', 'max_arm_torque_nm', 'max_joint_speed_rad_s']
    vectors = {'contact_point_m': 'contact', 'outgoing_velocity_mps': 'outgoing_velocity',
        'crossing_point_m': 'crossing', 'first_landing_m': 'first_landing'}
    exported = []
    for row in rows:
        outcome = independent_episode(row, protocol)
        flat = dict(record_scope='changing_policy_native_training' if learning else 'fixed_policy_TRAIN_assessment')
        flat.update({name: row[name] for name in common})
        flat.update({name: row[name] for name in FAULTS})
        flat.update({f'jump_{name}': row['jump'][name] for name in jump})
        flat.update(canonical_airborne_contact=outcome['airborne_contact'],
            canonical_conditional_recovery=outcome['conditional_recovery'],
            canonical_complete=outcome['complete'], canonical_rally_fault=outcome['fault'])
        for name, prefix in vectors.items():
            value = np.asarray(row[name], dtype=float)
            if value.shape != (3,):
                raise ValueError('Expected one native3D event vector')
            flat.update({prefix + '_' + axis: value[i] for i, axis in enumerate('xyz')})
        velocity = np.asarray(row['outgoing_velocity_mps'], dtype=float)
        flat['outgoing_ball_speed_mps'] = float(np.linalg.norm(velocity)) if np.isfinite(velocity).all() else None
        for name, value in flat.items():
            if isinstance(value, (bool, np.bool_)):
                flat[name] = int(value)
            elif value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value):
                flat[name] = ''
        exported.append(flat)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(exported[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(exported)
    table = list(csv.DictReader(io.StringIO(stream.getvalue())))
    assert len(table) == summary['episodes']
    assert [int(row['env_id']) for row in table] == [r['env_id'] for r in rows]
    for column, count in [('canonical_airborne_contact', summary['airborne_contacts']),
                          ('canonical_conditional_recovery', summary['conditional_recovery_count']),
                          ('canonical_complete', summary['complete'])]:
        assert sum(int(row[column]) for row in table) == count
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue())
    temporary.replace(output)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope=('Tabular export of changing-policy native TRAIN learning; no frozen performance.' if learning else
               'Tabular export of completed fixed-policy native TRAIN assessment; no new simulation or performance.'),
        source_run=str(run.relative_to(ROOT)),
        source_episode_sha256=sha(rows_path), protocol_sha256=sha(protocol_path),
        csv=str(output.relative_to(ROOT)), csv_sha256=sha(output), rows=len(table), columns=len(table[0]),
        canonical_summary=summary, original_runtime_metrics_match=True,
        source_exporter_sha256=sha(Path(__file__)),
        vector_units=dict(contact='m', outgoing_velocity='m/s', crossing='m', first_landing='m'),
        missing_numeric_event_fields='Empty CSV cells preserve unobserved native events; they are not zero-valued events.',
        formal_cases_read=False, formal_acceptance=False)
    manifest['final_exported_policy_sha256' if learning else 'assessed_policy_sha256'] = policy_sha
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    print(json.dumps({key: manifest[key] for key in ['csv', 'csv_sha256', 'rows', 'columns', 'original_runtime_metrics_match']}))


if __name__ == '__main__':
    main()
