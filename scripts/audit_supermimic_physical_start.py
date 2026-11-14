#!/usr/bin/env python3
"""Inventory existing physical starts; exit 2 if task memory is incomplete.

No Isaac, rollout, optimizer, state reconstruction or future-command replay.
Scanning all rows establishes coverage of the same missing-state issue; it
does not advance to model training or policy updates after a failure.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import torch

from quadsmash.rl.supermimic_imagination_contract_v1 import physical_start_issues, require_physical_start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest = json.loads((args.archive/'manifest.json').read_text())
    totals = Counter(); phase_counts = Counter(); first = None; inventory = None
    checked_chunks = []; ticks = transitions = 0
    for entry in manifest['chunks']:
        path = args.archive/entry['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Archive hash mismatch: ' + str(path))
        checked_chunks.append({'file': entry['file'], 'sha256': digest})
        for index, row in enumerate(torch.load(path, map_location='cpu', weights_only=True)['rows']):
            issues = physical_start_issues(row['before'], row['executor_before'])
            for issue in issues:
                totals[issue['code'] + ':' + issue.get('group', '')] += len(row['observation'])
            for phase, count in enumerate(torch.bincount(row['observation'][:, 770:775].argmax(-1), minlength=5)):
                phase_counts[phase] += int(count)
            if first is None and issues:
                first = dict(chunk=entry['file'], row=index,
                    control_tick=row['before']['control_tick'], issues=issues)
                try:
                    require_physical_start(row['before'], row['executor_before'])
                except ValueError:
                    (args.output/'first_failure_traceback.txt').write_text(traceback.format_exc())
                memory = row['executor_before']
                inventory = dict(counts={k: len(v) for k,v in memory.items()},
                    object_types=memory['object_types'], aliases=memory['aliases'], omitted=memory['omitted'],
                    tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in memory['tensors'].items()},
                    values_note='Values include immutable configuration and mutable cursors; counts are not a Markov certificate',
                    cursor_values={k: v for k,v in memory['values'].items() if k.endswith('plan_cursor')},
                    captured_state_keys=list(row['before']), captured_recovery_keys=list(row['before']['recovery']))
            ticks += 1; transitions += len(row['observation'])
    result = dict(status='BLOCKED_MISSING_CAUSAL_STATE' if first else 'INVENTORY_ONLY_PASSED',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
        archive=str(args.archive.resolve()), checked_chunks=checked_chunks, ticks=ticks,
        transitions=transitions, phase_counts=dict(phase_counts), issue_counts=dict(totals),
        first_failure=first, physical_model_calls=0, actor_updates=0, critic_updates=0,
        native_steps=0, new_task_success_rate=None,
        source_policy_sha256=manifest['source_sha256'], task_contract=manifest['task_contract'],
        limitation='No claim of executor replay, reward parity, gradient parity or native performance')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'execution_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status','ticks','transitions','phase_counts','issue_counts',
        'physical_model_calls','actor_updates','native_steps')}, indent=2), flush=True)
    return 2 if first else 0


if __name__ == '__main__':
    raise SystemExit(main())
