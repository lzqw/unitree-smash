#!/usr/bin/env python3
"""Export real complete episodes for BC, with an explicit speed curriculum.

The original independent score is retained. A lower requested speed changes
only the dataset label, never the simulator, evaluator, or original outcome.
Episode boundaries are retained for case-level splitting and sequence use.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row, protocol, minimum_speed):
    if not math.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError('Require an explicit positive curriculum speed')
    original = independent_episode(row, protocol)
    court = json.loads((Path(protocol['_root']) / 'configs/court_net_v4.json').read_text())
    v = np.asarray(row['outgoing_velocity_mps'])
    landing = np.asarray(row['first_landing_m'])
    contact, release, crossing, ground = [row[k] for k in (
        'contact_time_s', 'release_time_s', 'crossing_time_s', 'first_landing_time_s')]
    tolerance = protocol['physics_dt_s']
    criteria = dict(
        ground_single_launch=bool(row['ground_init'] and row['active_rally_launch_count'] == 1),
        legal_airborne_strings=original['airborne_contact'],
        same_flight_stable_recovery=original['conditional_recovery'],
        no_rule_fault=not original['fault'],
        released_downward_forward=bool(row['released'] and np.isfinite(v).all() and v[0] > 0 and v[2] <= -1.),
        actual_net_and_target=bool(row['crossed_net'] and row['landed_target'] and np.isfinite(landing).all()
            and court['target_x_limits_m'][0] <= landing[0] <= court['target_x_limits_m'][1]
            and abs(landing[1]) <= court['target_y_half_width_m']
            and court['opponent_x_limits_m'][0] <= landing[0] <= court['opponent_x_limits_m'][1]
            and abs(landing[1]) <= court['singles_width_m'] / 2),
        event_order=bool(np.isfinite([contact, release, crossing, ground]).all()
            and contact >= protocol['preparation_seconds'] - tolerance and release >= contact
            and crossing + tolerance >= contact and ground + tolerance >= max(crossing, release)),
        no_numerical_failure=row['numerical_failures'] == 0,
    )
    speed = float(np.linalg.norm(v)) if np.isfinite(v).all() else None
    all_other = all(criteria.values())
    if bool(all_other and speed is not None and speed >= 8.) != bool(original['complete'] and row['numerical_failures'] == 0):
        raise ValueError('Original8m/s endpoint disagreement')
    accepted = bool(all_other and speed is not None and speed >= minimum_speed)
    return dict(original_outcomes=original, original_complete=original['complete'],
        curriculum_complete=accepted, outgoing_speed_mps=speed,
        criteria_except_speed=criteria,
        rejection_reasons=[k for k, value in criteria.items() if not value]
            + ([] if speed is not None and speed >= minimum_speed else ['speed_threshold']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-evaluation', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--minimum-speed-mps', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior datasets; choose a new directory')
    source = args.source_evaluation.resolve()
    job = json.loads((source / 'job.json').read_text())
    if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or (source / 'error.txt').exists():
        raise ValueError('Require a successfully completed native source')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    inputs = json.loads((source / 'INPUTS.json').read_text())
    if inputs['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Source uses a different original task')
    raw_rows = json.loads((source / 'episodes.json').read_text())
    cases, indices = training_cases(ROOT, protocol)
    original_summary = summarize_training(raw_rows, cases, indices, protocol)
    if original_summary['numerical_failures']:
        raise ValueError('Numerical failures invalidate demonstration collection')
    by_index = {row['source_training_index']: row for row in raw_rows}
    manifest_path = args.traces / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['schema'] != 'fixed-policy-native-first-episode-trace-v1'
            or manifest['action_interface'] != 'absolute-joint-limits-v1'
            or manifest['control_dt_s'] != .01):
        raise ValueError('Require the original full-observation/action recording contract')
    provenance = manifest['provenance']
    old_data = None
    if provenance.get('native_execution') is True:
        if args.traces.resolve() != source / 'fixed_policy_traces':
            raise ValueError('Native recording must belong to this completed evaluation')
        if (provenance['protocol_sha256'] != sha(protocol_path)
                or provenance['policy_sha256'] != inputs['policy_sha256']
                or provenance['asset_sha256'] != protocol['asset']['sha256']):
            raise ValueError('Native trace source identity mismatch')
        limits = np.asarray(provenance['joint_limits_rad'], dtype=np.float32)
    else:
        # Older six-case recordings were losslessly repackaged on CPU. Match
        # their public observations AND dispatched actions to the actual file.
        if Path(provenance['source']).resolve() != source:
            raise ValueError('Repackaged recording refers to another source')
        old_path = source / 'public_and_actions_003.npz'
        if provenance['source_sha256'] != sha(old_path):
            raise ValueError('Original native public/action source changed')
        old_data = np.load(old_path)
        limits = np.asarray(inputs['joint_limits_rad'], dtype=np.float32)
    observations, actions, labels, offsets, seen = [], [], [], [0], set()
    for entry in manifest['episodes']:
        index = entry['source_training_index']
        if index in seen or index not in by_index:
            raise ValueError('Duplicate or unknown source case')
        seen.add(index)
        row = by_index[index]
        def canonical(value):
            return json.dumps({k: v for k, v in value.items() if k != 'env_id'}, sort_keys=True)
        if canonical(row) != canonical(entry['raw_episode']):
            raise ValueError('Recorded episode differs from the actual source result')
        outcome = classify(row, protocol, args.minimum_speed_mps)
        if outcome['original_outcomes'] != entry['original_outcomes']:
            raise ValueError('Original independent labels changed')
        path = args.traces / entry['data_file']
        if sha(path) != entry['data_sha256']:
            raise ValueError('Trace data integrity failure')
        with np.load(path) as data:
            public = unpack_public(data['packed_public'])
            dispatched = data['dispatched_actions'].copy()
        n = len(public)
        if (n != entry['samples'] or dispatched.shape != (n, 18)
                or not np.isfinite(dispatched).all() or np.abs(dispatched).max() > 1.000001
                or hashlib.sha256(public.tobytes()).hexdigest() != entry['full_public794_sha256']
                or np.max(abs(public[:, 775] - np.arange(n) * .01)) > 2e-6):
            raise ValueError('Incomplete, altered, or invalid observation/action sequence')
        if old_data is not None:
            lane = int(np.flatnonzero(old_data['training_indices'] == index)[0])
            valid = old_data['valid_first_episode'][:, lane]
            if (int(valid.sum()) != n or not valid[:n].all()
                    or not np.array_equal(public, old_data['observations'][:n, lane])
                    or not np.array_equal(dispatched, old_data['actions'][:n, lane])):
                raise ValueError('CPU serialization differs from actual native inputs or actions')
        label = dict(training_index=index, case_id=row['case_id'], source_trace_sha256=sha(path),
            samples=n, **outcome)
        if outcome['curriculum_complete']:
            label['dataset_episode_index'] = len(observations)
            observations.append(public); actions.append(dispatched); offsets.append(offsets[-1] + n)
        labels.append(label)
    if old_data is not None:
        old_data.close()
    obs = np.concatenate(observations) if observations else np.empty((0, 794), dtype=np.float32)
    act = np.concatenate(actions) if actions else np.empty((0, 18), dtype=np.float32)
    target = limits[:, 0] + .5 * (act + 1.) * (limits[:, 1] - limits[:, 0])
    args.output.mkdir(parents=True)
    data_path = args.output / 'demonstrations.npz'
    np.savez_compressed(data_path, observations=obs, actions_normalized=act,
        joint_targets_rad=target, joint_limits_rad=limits, episode_offsets=np.asarray(offsets, dtype=np.int64))
    result = dict(schema='complete-native-demonstrations-v1', source_evaluation=str(source),
        source_episodes_sha256=sha(source / 'episodes.json'), trace_manifest_sha256=sha(manifest_path),
        protocol_sha256=sha(protocol_path), asset_sha256=protocol['asset']['sha256'],
        original_task_speed_mps=8., curriculum_minimum_speed_mps=args.minimum_speed_mps,
        only_speed_condition_varied=True, original_source_summary=original_summary,
        available_trace_episodes=len(labels), accepted_episodes=len(observations), samples=len(obs),
        accepted_training_indices=[a['training_index'] for a in labels if a['curriculum_complete']],
        accepted_original_complete_episodes=sum(a['original_complete'] for a in labels if a['curriculum_complete']),
        data_file=data_path.name, data_sha256=sha(data_path), episodes=labels,
        scope='Offline dataset from previous actual fixed-policy episodes; no new physical execution or learned-policy performance.',
        action_interface=manifest['action_interface'], control_dt_s=.01,
        split_requirement='Split by source case/episode, never randomly split adjacent frames.',
        formal_acceptance=False, goal_complete=False)
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['accepted_episodes', 'samples', 'accepted_training_indices',
        'accepted_original_complete_episodes', 'curriculum_minimum_speed_mps', 'data_sha256']}, indent=2))


if __name__ == '__main__':
    main()
