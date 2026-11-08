"""Reproduce the native tick550 representation mismatch and check its repair.

Queries, poses and the accepted plan come from completed/partial native
evidence. This recomputes commands only; it is not another physical rollout.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pinocchio as pin
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_current_state_recovery_v2 import NativeCurrentStateRecoveryV2, canonical_state
from scripts.loaded_pinocchio_model_v1 import rotation_wxyz
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['bundle', 'partial-run', 'paired-run', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier checks')
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    calls = json.loads((args.partial_run / 'current_state_planning.json').read_text())['calls']
    if [c['control_tick'] for c in calls] != [400, 550] or not all(c['accepted'] for c in calls):
        raise ValueError('Expected two actually accepted native planning queries')
    job = json.loads((args.paired_run / 'job.json').read_text())
    if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0:
        raise ValueError('Completed paired trajectory required')
    with np.load(args.paired_run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    with np.load(args.paired_run / 'all_first_episodes.npz', allow_pickle=False) as z:
        times = z['episode_clock_s'][:, 3].copy()
        poses = z['link_poses'][:, 3, 0].copy()
    rows = []
    with tempfile.TemporaryDirectory(prefix='recovery-representation-') as temporary:
        adapter = NativeCurrentStateRecoveryV2(args.bundle, 3,
            torch.tensor(manifest['configuration']['interface']['limits']), manifest['policy_sha256'], temporary)
        for call in calls:
            tick = call['control_tick']; i = tick - 280
            if not all(np.array_equal(call['query'][k], trace[t][i]) for k, t in
                    [('q', 'q'), ('v', 'v'), ('contacts', 'contacts'), ('target', 'previous_target_rad')]):
                raise ValueError('Paired trajectory does not reproduce the actual planning query')
            state = state_from_arrays(adapter.model, trace['q'][i], trace['v'][i],
                trace['contacts'][i], trace['foot_force_world_n'][i])
            state.previous_command_target = trace['previous_target_rad'][i].copy()
            j = int(np.argmin(abs(times - .01 * tick)))
            if abs(times[j] - .01 * tick) > 1e-6:
                raise ValueError('No exact native pose sample at the planning instant')
            rotation = rotation_wxyz(poses[j, 3:])
            if not np.array_equal(pin.Quaternion(rotation).coeffs(), state.q[3:7]):
                raise ValueError('Native matrix does not reproduce recorded model quaternion')
            state.root_rotation_world = rotation
            feedback = adapter.feedback_base
            for patch in call['plan']['corrections']:
                feedback = SuffixCommandCorrectionV1(feedback, patch['control_tick'],
                    patch['previous_command'], patch['knots_rad'])
            old, _ = feedback.action(tick - 200, state, state.previous_command_target)
            corrected, errors = canonical_state(state, adapter.model)
            new, _ = feedback.action(tick - 200, corrected, corrected.previous_command_target)
            if not np.array_equal(new, call['first_command']):
                raise ValueError('Canonicalized native command still differs from accepted prediction')
            bad = copy.deepcopy(state); bad.root_rotation_world[0, 0] += 1e-6
            rejected = False
            try:
                canonical_state(bad, adapter.model)
            except ValueError:
                rejected = True
            if not rejected:
                raise ValueError('Contradictory measured state silently accepted')
            rows.append(dict(control_tick=tick, query_exactly_matches_paired_native_trace=True,
                original_command_exact=bool(np.array_equal(old, call['first_command'])),
                original_command_error_rad=float(np.max(abs(old - call['first_command']))),
                redundant_field_errors=errors, corrected_command_exact=True,
                corrected_command_error_rad=float(np.max(abs(new - call['first_command']))),
                measured_state_and_history_preserved=True, contradictory_state_rejected=True))
        adapter.close()
    if rows[0]['original_command_error_rad'] != 0 or rows[1]['original_command_error_rad'] == 0:
        raise ValueError('Historical tick550 failure was not reproduced')
    paths = [Path(__file__), args.bundle / 'manifest.json', args.partial_run / 'current_state_planning.json',
        args.partial_run / 'error.txt', args.paired_run / 'sequence_native_trace.npz',
        args.paired_run / 'all_first_episodes.npz']
    paths += [Path(m.__file__) for n, m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(scope=__doc__, rows=rows, source_sha256=pins, native_execution=False,
        exact_command_check_retained=True, physical_or_task_thresholds_changed=False, valid_expert_labels=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
