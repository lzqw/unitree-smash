"""Necessary leg-command representability bound on a completed native trace.

For a fixed base command, the clamped residual mixer can move a leg target
by at most 2*scale relative to any other bounded residual. Exceeding that
bound proves this recorded teacher target cannot be exactly reproduced by
changing only the residual. Passing it does not prove representability.
Arm commands are excluded because their limiter has history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control import full_racket_residual_v4 as mixer_module


def audit(run, checkpoint):
    run, checkpoint = Path(run), Path(checkpoint)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files = [Path(__file__), Path(mixer_module.__file__), checkpoint]
    files += [run / n for n in ['job.json', 'native_command_inputs.json',
        'sequence_native_result.json', 'sequence_native_trace.npz', 'sequence_dispatch.npz']]
    pins = {str(p.resolve()): sha(p) for p in files}
    job = json.loads((run / 'job.json').read_text())
    if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0:
        raise ValueError('Completed native execution required; audit admission is separate')
    commands = json.loads((run / 'native_command_inputs.json').read_text())
    if sha(checkpoint) != commands['policy_sha256']:
        raise ValueError('Checkpoint differs from executed student')
    # Explicit trusted local checkpoint, as used by the native evaluator.
    policy = torch.load(checkpoint, map_location='cpu', weights_only=False)
    scale = np.asarray(policy['residual_joint_scale_rad'], dtype=np.float64)
    limits = np.asarray(commands['joint_limits_rad'], dtype=np.float64)
    if scale.shape != (18,) or limits.shape != (18, 2) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid recorded residual interface')
    meta = json.loads((run / 'sequence_native_result.json').read_text())
    ticks = np.asarray([r['control_tick'] for r in meta['records']])
    if not np.array_equal(ticks, np.arange(meta['start_tick'], meta['start_tick'] + len(ticks))):
        raise ValueError('Continuous original-episode command trace required')
    with np.load(run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        student = z['student_action'].copy()
        final_action = z['final_action'].copy()
        target = z['final_target_rad'].copy()
    with np.load(run / 'sequence_dispatch.npz', allow_pickle=False) as z:
        if not np.array_equal(student, z['student'][ticks, 3]) or not np.array_equal(final_action, z['final'][ticks, 3]):
            raise ValueError('Trace does not match actual TRAIN3 dispatch')
        if not z['first_episode'][ticks, 3].all():
            raise ValueError('Trace spans a reset')
    decode = lambda x: limits[:, 0] + .5 * (x + 1.) * (limits[:, 1] - limits[:, 0])
    if target.shape != (len(ticks), 18) or not np.isfinite(target).all():
        raise ValueError('Invalid native targets')
    decode_error = float(np.max(np.abs(decode(final_action) - target)))
    if decode_error > 1e-6:
        raise ValueError('Recorded target differs from final normalized command')
    delta = np.abs(target[:, :12] - decode(student)[:, :12])
    lower = np.maximum(delta - 2 * scale[:12], 0.)
    violations = lower > 1e-6
    rows = [dict(control_tick=int(ticks[i]), joint_index=int(j),
        command_difference_rad=float(delta[i, j]), two_residual_scales_rad=float(2 * scale[j]),
        unavoidable_error_lower_bound_rad=float(lower[i, j])) for i, j in np.argwhere(violations)]
    result = dict(scope=__doc__, native_run=str(run.resolve()), source_sha256=pins,
        control_steps=len(ticks), residual_joint_scale_rad=scale.tolist(),
        maximum_action_decode_error_rad=decode_error,
        steps_proven_outside_fixed_base_residual_authority=int(violations.any(1).sum()),
        maximum_leg_command_difference_rad=float(delta.max()),
        maximum_unavoidable_command_error_lower_bound_rad=float(lower.max()),
        violations=rows, native_rerun=False, expert_labels_created=False,
        limitations=['Only legs and the same recorded base-controller output are considered.',
            'Within this conservative bound does not prove a feasible residual label exists.',
            'Different learned actions may recover without matching these teacher commands.',
            'This does not establish physical safety, robust recovery, or teacher admission.'])
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Input changed during audit: ' + path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    result = audit(args.run, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'violations')}, indent=2))


if __name__ == '__main__':
    main()
