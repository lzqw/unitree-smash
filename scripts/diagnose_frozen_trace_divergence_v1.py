"""Locate observation/action divergence in immutable, ground-start native traces.

This is temporal evidence, not a proof of physical isolation or a replacement
for the existing native audit. No tolerances or expert labels are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(run, case):
    manifest_path = run / 'fixed_policy_traces/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    matches = [r for r in manifest['episodes'] if r['source_training_index'] == case]
    if len(matches) != 1:
        raise ValueError('Require one first-episode trace for the requested case')
    entry = matches[0]
    path = manifest_path.parent / entry['data_file']
    if sha(path) != entry['data_sha256']:
        raise ValueError('Trace differs from its immutable manifest')
    with np.load(path, allow_pickle=False) as z:
        public = unpack_public(z['packed_public'])
        action = z['dispatched_actions'].copy()
    if len(public) != entry['samples'] or action.shape != (len(public), 18):
        raise ValueError('Incomplete or incompatible native trace')
    if not np.isfinite(public).all() or not np.isfinite(action).all():
        raise ValueError('Nonfinite trajectory')
    if not np.allclose(public[:, 775], np.arange(len(public)) * .01, atol=2e-6, rtol=0):
        raise ValueError('Trace is not a continuous 100 Hz ground-start episode')
    return public, action, {str(path.resolve()): sha(path),
                            str(manifest_path.resolve()): sha(manifest_path)}


def first_above(errors, threshold):
    indices = np.flatnonzero(errors > threshold)
    return int(indices[0]) if len(indices) else None


def compare(baseline, run, case):
    before, old_action, files = load_trace(baseline, case)
    after, action, more = load_trace(run, case)
    files.update(more)
    n = min(len(before), len(after))
    before, after = before[:n], after[:n]
    old_action, action = old_action[:n], action[:n]
    observation_error = np.max(abs(after - before), axis=1)
    action_error = np.max(abs(action - old_action), axis=1)
    tick = first_above(observation_error, 0.)
    result = dict(case=case, compared_samples=n,
                  maximum_public_error=float(observation_error.max()),
                  maximum_action_error=float(action_error.max()),
                  first_observation_difference_tick=tick,
                  first_action_difference_tick=first_above(action_error, 0.),
                  first_observation_over_existing_audit_tolerance_tick=first_above(observation_error, 5e-4),
                  first_action_over_existing_audit_tolerance_tick=first_above(action_error, 5e-6))
    if tick is not None:
        # Observation at tick t precedes action t. The action at t-1 produced
        # the preceding interval; action t may legitimately respond to a changed state.
        groups = {'base_linear_velocity': (0, 3), 'base_angular_velocity': (3, 6),
                  'projected_gravity': (6, 9), 'joint_position_public': (9, 27),
                  'joint_velocity_public': (27, 45), 'foot_contact_flags': (57, 61),
                  'ball_public': (61, 69), 'observation_history': (69, 759),
                  'previous_action_public': (776, 794)}
        result['first_observation_difference'] = dict(
            time_s=float(after[tick, 775]),
            all_preceding_dispatched_actions_bitwise_equal=bool(np.array_equal(action[:tick], old_action[:tick])),
            group_max_absolute_error={k: float(np.max(abs(after[tick, lo:hi]-before[tick, lo:hi])))
                                      for k, (lo, hi) in groups.items()},
            current_action_error=float(action_error[tick]),
            preceding_action_error=None if tick == 0 else float(action_error[tick-1]))
    return result, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier evidence')
    rows = []
    files = {str(Path(__file__).resolve()): sha(Path(__file__))}
    for run in args.run:
        row, sources = compare(args.baseline, run, args.case)
        rows.append(dict(run=str(run.resolve()), **row))
        files.update(sources)
    result = dict(scope=__doc__, comparisons=rows, new_native_execution=False,
                  valid_expert_labels=False, audit_tolerances_changed=False,
                  cause_identified=False, source_sha256=files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
