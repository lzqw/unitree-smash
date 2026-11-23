#!/usr/bin/env python3
"""Evaluate a chosen frozen checkpoint on 512 predeclared launches at the original native timebase."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile

from compare_supermimic_fixed_stages import load_run
from quadsmash.rl.supermimic_independent_panel import SHA256, load_panel, panel_batch
from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING


def validated_batch(batch, i, checkpoint_sha, panel, *, approach_ball_only=False, early_arm_seconds=None):
    scores, identity, actor = load_run(batch)
    expected, suite = panel_batch(i, panel['protocol_sha256'], identity['seed'])
    metrics = json.loads((batch/'metrics.json').read_text())
    if (metrics['evaluated_checkpoint_sha256'] != checkpoint_sha
            or identity['task'].get('independent_evaluation') != suite
            or identity['task']['protocol_sha256'] != panel['protocol_sha256']
            or sorted((r['case_id'], lane) for lane,r in enumerate(expected)) != identity['assignments']):
        raise ValueError(f'Independent batch {i} has a different checkpoint, protocol or case assignment')
    if approach_ball_only and (identity['scoring'] != APPROACH_BALL_SCORING or not all(
            row.get('scoring_schema') == APPROACH_BALL_SCORING and row.get('episode_index') == 0
            for row in scores)):
        raise ValueError(f'Independent batch {i} has different ball-only episode scoring')
    if early_arm_seconds is not None and identity['execution'].get('arm_preparation_seconds') != early_arm_seconds:
        raise ValueError(f'Independent batch {i} has a different A0 preparation gate')
    return scores, identity, actor


def collect_scores(folder, checkpoint_sha, panel, *, approach_ball_only=False, early_arm_seconds=None):
    rows = []
    first_contract = None
    for i in range(4):
        scores, identity, actor = validated_batch(folder/f'batch_{i}', i, checkpoint_sha, panel,
            approach_ball_only=approach_ball_only, early_arm_seconds=early_arm_seconds)
        task = {key: value for key, value in identity['task'].items() if key != 'independent_evaluation'}
        contract = (actor, identity['execution'], identity['physics'], identity['scoring'], task)
        if first_contract is not None and contract != first_contract:
            raise ValueError('Policy, physics or scoring differ across independent batches')
        first_contract = contract
        rows.extend(scores)
    if len({r['case_id'] for r in rows}) != 512:
        raise ValueError('Missing or repeated independent episodes')
    return dict(panel_sha256=SHA256, checkpoint_sha256=checkpoint_sha, actor_sha256=actor,
                protocol_sha256=panel['protocol_sha256'], scoring_schema=identity['scoring'], episodes=len(rows),
                numerical_failures=sum(r['numerical_failures'] for r in rows),
                metrics=summarize_native_episodes(rows), fixed_updates=0)


def compare_ball_only(baseline, candidate, baseline_sha, candidate_sha, panel):
    gains, regressions, speed_changes = [], [], []
    success = 'ball_side_complete_except_recovery'
    counts = {name: {'success': 0, 'qualified_contacts': 0, 'low_speed_after_qualified_contact': 0}
              for name in ('baseline', 'candidate')}
    for i in range(4):
        folders = (baseline/f'batch_{i}', candidate/f'batch_{i}')
        (a, contract_before, _), (b, contract_after, _) = (
            validated_batch(path, i, sha, panel, approach_ball_only=True)
            for path, sha in zip(folders, (baseline_sha, candidate_sha), strict=True))
        actuator_before, actuator_after = (json.loads((path/'actuator_contract.json').read_text()) for path in folders)
        if contract_before != contract_after or actuator_before != actuator_after:
            raise ValueError(f'Independent batch {i} changed execution, initial states or actuator contract')
        baseline_rows, candidate_rows = ({row['case_id']: row for row in scores} for scores in (a, b))
        for identity, before in baseline_rows.items():
            after = candidate_rows[identity]
            if before[success] != after[success]:
                (gains if after[success] else regressions).append(identity)
            for name, row in (('baseline', before), ('candidate', after)):
                qualified = row['qualified_airborne_string_contact']
                counts[name]['success'] += int(row[success])
                counts[name]['qualified_contacts'] += int(qualified)
                counts[name]['low_speed_after_qualified_contact'] += int(qualified
                    and row['outgoing_speed_mps'] is not None and 'speed_threshold' in row['rejection_reasons'])
            if all(row['qualified_airborne_string_contact'] and row['outgoing_speed_mps'] is not None
                   for row in (before, after)):
                speed_changes.append(after['outgoing_speed_mps'] - before['outgoing_speed_mps'])
    flips = len(gains) + len(regressions)
    probability = min(1., 2 * sum(math.comb(flips, i) for i in range(min(len(gains), len(regressions)) + 1)) / 2**flips)
    return dict(cases=512, baseline_checkpoint_sha256=baseline_sha, **counts, improved_cases=gains,
                regressed_cases=regressions, paired_exact_two_sided_p=probability,
                paired_qualified_release_speed_cases=len(speed_changes),
                mean_paired_release_speed_delta_mps=mean(speed_changes) if speed_changes else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','g07-execution-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--approach-ball-only', action='store_true')
    parser.add_argument('--early-arm-seconds', type=float, help='Use the registered early-arm gate for A0')
    parser.add_argument('--paired-baseline', type=Path, help='Completed frozen 512-case baseline under the same gate')
    args = parser.parse_args()
    if not args.approach_ball_only and (args.early_arm_seconds is not None or args.paired_baseline is not None):
        parser.error('The A0 gate override and paired baseline require ball-only scoring')
    folder = args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    panel = load_panel()
    with (folder/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with args.checkpoint.open('rb') as stream:
            checkpoint_sha = hashlib.file_digest(stream,'sha256').hexdigest()
        expectations = dict(approach_ball_only=args.approach_ball_only, early_arm_seconds=args.early_arm_seconds)
        baseline_sha = None
        if args.paired_baseline is not None:
            baseline_summary = json.loads((args.paired_baseline/'independent512.json').read_text())
            baseline_sha = baseline_summary['checkpoint_sha256']
            actual = collect_scores(args.paired_baseline, baseline_sha, panel, approach_ball_only=True)
            if any(baseline_summary.get(key) != actual[key] for key in
                   ('panel_sha256', 'actor_sha256', 'episodes', 'metrics')):
                raise ValueError('Paired baseline summary disagrees with its 512 frozen episodes')
        for i in range(4):
            batch = folder/f'batch_{i}'
            if batch.is_dir():
                validated_batch(batch, i, checkpoint_sha, panel, **expectations)
                continue
            with tempfile.TemporaryDirectory(dir=folder,prefix=f'.batch_{i}_') as temp:
                run_dir = Path(temp)/'run'
                cmd = [sys.executable,'scripts/evaluate_supermimic_v1.py','--checkpoint',str(args.checkpoint.resolve()),
                       '--g07-execution-root',str(args.g07_execution_root.resolve()),'--output',str(run_dir),
                       '--seed',str(panel['native_reset_seed']),'--independent-panel-batch',str(i),
                       '--device','cuda:0','--physics-device','cuda:0','--headless','--kit_args=--renderer/multiGpu/enabled=false']
                if args.approach_ball_only:
                    cmd.append('--approach-ball-only')
                if args.early_arm_seconds is not None:
                    cmd.extend(('--early-arm-seconds', str(args.early_arm_seconds)))
                with (folder/f'batch_{i}.log').open('a') as log:
                    subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,PYTHONPATH='.:scripts'),
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
                validated_batch(run_dir, i, checkpoint_sha, panel, **expectations)
                run_dir.rename(batch)
        result = collect_scores(folder, checkpoint_sha, panel, **expectations)
        if baseline_sha is not None:
            result['paired_ball_only'] = compare_ball_only(args.paired_baseline, folder, baseline_sha, checkpoint_sha, panel)
        temp = folder/'independent512.json.tmp'
        temp.write_text(json.dumps(result,indent=2)+'\n')
        temp.replace(folder/'independent512.json')
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
