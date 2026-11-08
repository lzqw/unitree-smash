"""Validate query-bank completeness and exact original-student replay.

Passing establishes passive recorded planning inputs, not recoverability,
native reset capability, or valid expert actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadsmash.rl.episode_trace_v4 import unpack_public


def audit(run, baseline):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    pins = {}
    for folder in [run, baseline]:
        job = read(folder / 'job.json')
        if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0:
            raise ValueError('Completed native executions required')
        pins[str(folder / 'job.json')] = sha(folder / 'job.json')
    meta_path = run / 'recovery_query_bank.json'
    data_path = run / 'recovery_query_bank.npz'
    meta = read(meta_path)
    if sha(data_path) != meta['data_sha256'] or not meta['complete_window']:
        raise ValueError('Incomplete or changed bank')
    if len(meta['direct_adapter_checks']) != 12:
        raise ValueError('Missing direct native-adapter checks')
    expected_checks = {(t, lane) for t in [280, 350, 550, 750] for lane in [3, 58, 125]}
    if {(r['control_tick'], r['lane']) for r in meta['direct_adapter_checks']} != expected_checks:
        raise ValueError('Unexpected direct-check coverage')
    if any(max(r['errors'].values()) > 1e-12 or not r['contacts_exact'] for r in meta['direct_adapter_checks']):
        raise ValueError('Direct adapter inconsistency')
    rows = []
    with np.load(data_path, allow_pickle=False) as z:
        ticks = z['control_ticks']
        if not np.array_equal(ticks, np.arange(200, 800)):
            raise ValueError('Unexpected control times')
        for key in z.files:
            if not np.isfinite(z[key]).all():
                raise ValueError('Nonfinite bank: ' + key)
            if key != 'control_ticks' and z[key].shape[:2] != (600, 128):
                raise ValueError('Unexpected bank dimensions: ' + key)
        for lane in range(128):
            name = f'fixed_policy_traces/case_{lane:03d}.npz'
            for folder in [run, baseline]:
                pins[str(folder / name)] = sha(folder / name)
            with np.load(run / name, allow_pickle=False) as a, np.load(baseline / name, allow_pickle=False) as b:
                public = unpack_public(a['packed_public'])
                same = all(np.array_equal(a[k], b[k]) for k in ['packed_public', 'dispatched_actions'])
                valid = z['unfinished_first_episode'][:, lane].astype(bool)
                selected = ticks[valid]
                if len(selected) and selected.max() >= len(a['dispatched_actions']):
                    raise ValueError('First-episode mapping exceeds native trace')
                action_exact = np.array_equal(z['student_action'][valid, lane], a['dispatched_actions'][selected])
                phase_exact = np.array_equal(z['phase'][valid, lane], public[selected, 770:775])
                clock_exact = np.array_equal(z['episode_clock'][valid, lane], public[selected, 775])
                rows.append(dict(lane=lane, observer_exact_replay=same,
                    first_episode_samples=int(valid.sum()), student_action_exact=action_exact,
                    phase_exact=phase_exact, clock_exact=clock_exact))
    for path in [meta_path, data_path, Path(__file__).resolve()]:
        pins[str(path)] = sha(path)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Evidence changed during audit')
    passed = all(all(row[k] for k in ['observer_exact_replay', 'student_action_exact',
        'phase_exact', 'clock_exact']) for row in rows)
    return dict(passed=passed, cases=rows, source_sha256=pins,
        complete_capture=True, direct_native_checks=meta['direct_adapter_checks'],
        valid_expert_labels=False, native_reset_capable=False,
        scope=__doc__)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['run', 'baseline', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous audit')
    result = audit(args.run.resolve(), args.baseline.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['passed', 'complete_capture', 'valid_expert_labels']}))
    if not result['passed']:
        raise SystemExit(2)
