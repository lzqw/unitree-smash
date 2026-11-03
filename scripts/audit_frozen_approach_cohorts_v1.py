#!/usr/bin/env python3
"""Compare completed native evaluations by the baseline's observed failures.

This is an offline audit. It never changes the controller, task, or GPU queue.
The cohorts are assigned from baseline events before reading candidate scores.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_frozen_complete5_v2 import compare, read_run, sha


def event_stages(source, summary):
    rows = json.loads((Path(source) / 'episodes.json').read_text())
    result = {s['training_index']: dict(s) for s in summary['stages']}
    if len(result) != 128 or {r['source_training_index'] for r in rows} != set(result):
        raise ValueError('Require one measured episode per original development condition')
    for row in rows:
        result[row['source_training_index']].update(
            racket_contact=bool(row['had_contact']), strings_contact=bool(row['string_hit']),
            frame_contact=bool(row['frame_hit']))
    return result


def audit(before_path, after_path):
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(ROOT)
    before, initial_before = read_run(before_path, protocol)
    after, initial_after = read_run(after_path, protocol)
    initial_errors = {}
    if len(initial_before) != 128 or len(initial_after) != 128:
        raise ValueError('Require all paired native initial states')
    # read_run verifies the same ordered training_indices. Compare every state,
    # rather than using identical seeds as a substitute for matched conditions.
    for field in ['root_state_local', 'joint_position_rad', 'joint_velocity_rad_s']:
        error = max(float(np.abs(np.asarray(a[field]) - np.asarray(b[field])).max())
                    for a, b in zip(initial_before, initial_after))
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError('Paired initial states differ: ' + field)
        initial_errors[field] = error
    old, new = event_stages(before_path, before), event_stages(after_path, after)
    cohorts = {
        'old_complete5': [i for i, s in old.items() if s['complete5']],
        'racket_contact_without_strings': [i for i, s in old.items()
                                          if not s['air'] and s['racket_contact'] and not s['strings_contact']],
        'no_racket_contact': [i for i, s in old.items() if not s['air'] and not s['racket_contact']],
        'strings_without_qualified_air': [i for i, s in old.items() if not s['air'] and s['strings_contact']],
    }
    failure_members = sum((cohorts[k] for k in cohorts if k != 'old_complete5'), [])
    if len(failure_members) != len(set(failure_members)) or set(failure_members) != {i for i, s in old.items() if not s['air']}:
        raise ValueError('Failure cohorts must partition baseline unqualified-air cases')
    fields = ['racket_contact', 'strings_contact', 'air', 'ball5_except_recovery', 'recovery', 'complete5']
    results = {}
    for name, members in cohorts.items():
        outcomes = {field: sorted(i for i in members if new[i][field]) for field in fields}
        reasons = Counter(reason for i in members if not new[i]['complete5']
                          for reason in new[i]['rejection_reasons'])
        results[name] = dict(total=len(members), baseline_ids=sorted(members),
                             candidate_counts={k: len(v) for k, v in outcomes.items()},
                             candidate_case_ids=outcomes,
                             candidate_rejection_reason_counts=dict(reasons))
    return dict(schema='native-frozen-approach-baseline-cohorts-v1',
                before={k: v for k, v in before.items() if k != 'stages'},
                after={k: v for k, v in after.items() if k != 'stages'},
                paired_initial_max_error=initial_errors,
                **compare(before, after), baseline_cohorts=results,
                rejection_counts_are_nonexclusive=True,
                formal_acceptance=False, generalization_test=False,
                scope='Original fixed128 TRAIN development cases; complete5 and complete8 separate; '
                      'same baseline controller and persistent recovery must be interpreted separately from full independent RL.')


def write_new(path, result):
    if path.exists():
        raise ValueError('Preserve earlier audit: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(path)


def follow_pipeline(state_path, before_path, output):
    """Watch an identified existing coordinator; never launch a GPU job."""
    state_path = state_path.resolve()
    if output.exists():
        raise ValueError('Use a fresh follower directory')
    state = json.loads(state_path.read_text())
    baseline_inputs = json.loads((before_path / 'INPUTS.json').read_text())
    if sha(state['source_policy']) != baseline_inputs['policy_sha256']:
        raise ValueError('Pipeline source is not the baseline evaluated policy')
    coordinator_pid = int(state['pid'])
    stat_path = Path(f'/proc/{coordinator_pid}/stat')
    stat_fields = stat_path.read_text().rsplit(') ', 1)[1].split()
    start_ticks = stat_fields[19]
    output.mkdir(parents=True)
    report = dict(pid=os.getpid(), coordinator_pid=coordinator_pid,
                  coordinator_start_ticks=start_ticks, pipeline_state=str(state_path),
                  status='WAITING_FOR_FROZEN_EVALUATIONS', audits=[], goal_complete=False)

    def save(**changes):
        report.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        temp = output / 'state.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(output / 'state.json')

    deadline = time.monotonic() + 7 * 3600
    save()
    try:
        while time.monotonic() < deadline:
            state = json.loads(state_path.read_text())
            for paired_path in sorted(state_path.parent.glob('paired_seed*.json')):
                destination = output / (paired_path.stem + '_cohorts.json')
                if str(destination) in report['audits']:
                    continue
                paired = json.loads(paired_path.read_text())
                result = audit(before_path, Path(paired['after']['output']))
                write_new(destination, result)
                report['audits'].append(str(destination))
                save()
            if state['status'] in ['EXPERIMENT_FINISHED', 'FAILED']:
                save(status='FINISHED' if state['status'] == 'EXPERIMENT_FINISHED' else 'SOURCE_PIPELINE_FAILED')
                return
            fields = stat_path.read_text().rsplit(') ', 1)[1].split()
            if fields[0] == 'Z' or fields[19] != start_ticks:
                raise RuntimeError('Identified coordinator exited without a final pipeline state')
            time.sleep(5)
        raise TimeoutError('Read-only audit follower reached its seven-hour observation cap')
    except BaseException as error:
        save(status='FAILED', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--after', type=Path)
    mode.add_argument('--follow-pipeline', type=Path, help='Existing coordinator state.json; output is a fresh directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.follow_pipeline:
        follow_pipeline(args.follow_pipeline, args.before, args.output)
        return
    if args.output.exists():
        raise ValueError('Use a new audit path to preserve earlier evidence')
    result = audit(args.before, args.after)
    write_new(args.output, result)
    print(json.dumps(dict(before=result['before']['complete5'], after=result['after']['complete5'],
                          cohorts={k: v['candidate_counts'] for k, v in result['baseline_cohorts'].items()})))


if __name__ == '__main__':
    main()
