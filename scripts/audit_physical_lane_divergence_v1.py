"""Distinguish physical pose changes from observation-frame changes.

Reads completed original-panel runs only. This does not identify collision
partners or prove a causal source of cross-lane divergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_frozen_trace_divergence_v1 import compare, load_trace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical(run, case):
    job = json.loads((run/'job.json').read_text())
    if job['status'] != 'SUCCEEDED' or job['charged_env_steps'] != 102400:
        raise ValueError('Require a completed original 128-case native run')
    path = run/'all_first_episodes.npz'
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key][:, case].copy() for key in z.files}
    clocks = values['episode_clock_s']
    if (clocks.shape != (400,) or not values['valid_first_episode'].all()
            or not np.allclose(clocks, np.arange(400)*.02, atol=2e-6, rtol=0)):
        raise ValueError('Require continuous first-episode 50 Hz physical poses')
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite physical trace')
    files = [path, run/'job.json', run/'geometry.json', run/'origin_layout.json']
    return values, {str(p.resolve()): sha(p) for p in files}


def physical_comparison(before, after):
    result = {}
    for key in ['link_poses', 'ball_position_m']:
        if before[key].shape != after[key].shape:
            raise ValueError('Changed physical trace layout')
        error = np.max(abs(after[key]-before[key]).reshape(400, -1), axis=1)
        differing = np.flatnonzero(error > 0)
        first = int(differing[0]) if len(differing) else None
        result[key] = dict(maximum_absolute_error=float(error.max()),
            bitwise_equal=bool(np.array_equal(before[key], after[key])),
            first_different_control_tick=None if first is None else 2*first)
        if key == 'link_poses':
            result[key]['maximum_position_error_m'] = float(np.max(abs(after[key][..., :3]-before[key][..., :3])))
            result[key]['maximum_quaternion_component_error'] = float(np.max(abs(after[key][..., 3:]-before[key][..., 3:])))
    return result


def audit(baseline, runs, case):
    initial, files = physical(baseline, case)
    baseline_geometry = json.loads((baseline/'geometry.json').read_text())
    baseline_origins = json.loads((baseline/'origin_layout.json').read_text())
    rows = []
    for run in runs:
        current, more = physical(run, case); files.update(more)
        if (json.loads((run/'geometry.json').read_text()) != baseline_geometry
                or json.loads((run/'origin_layout.json').read_text()) != baseline_origins):
            raise ValueError('Changed geometry or environment origins')
        public, more = compare(baseline, run, case); files.update(more)
        old_obs, _, _ = load_trace(baseline, case)
        obs, _, _ = load_trace(run, case)
        tick = public['first_observation_difference_tick']
        changed = [] if tick is None else np.flatnonzero(obs[tick] != old_obs[tick]).tolist()
        rows.append(dict(run=str(run.resolve()), public_and_action=public,
            physical=physical_comparison(initial, current),
            first_changed_public_indices=changed))
    files[str(Path(__file__).resolve())] = sha(__file__)
    for relative in ['scripts/diagnose_frozen_trace_divergence_v1.py', 'quadsmash/rl/episode_trace_v4.py']:
        files[str(ROOT/relative)] = sha(ROOT/relative)
    return dict(case=case, comparisons=rows, source_sha256=files,
        new_native_execution=False, cause_identified=False,
        physical_sampling_hz=50, public_sampling_hz=100,
        audit_tolerances_changed=False, valid_expert_labels=False,
        interpretation='Relative ball observations may change with robot motion while the world ball trajectory is unchanged. Physical pose samples do not resolve the first differing physics substep.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    result = audit(args.baseline, args.run, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['comparisons'], indent=2))
