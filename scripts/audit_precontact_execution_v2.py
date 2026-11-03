#!/usr/bin/env python3
"""Replay measured commands to audit slew limiting before racket contact.

The replay is a CPU consistency check on recorded inputs, not a physical
counterfactual. Joint error is target minus measured angle at command issue;
it does not establish that tracking error caused an unsuccessful contact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.build_verified_fixed_retention_v2 as builder
from scripts.compare_frozen_complete5_v2 import read_run, sha
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.frozen_v2_approach_v1 import PublicAerialProximityGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audit')
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
    protocol['_root'] = str(ROOT)
    summary, _ = read_run(args.evaluation, protocol)
    if summary['policy_sha256'] != sha(args.policy):
        raise ValueError('Policy differs from actual evaluation source')
    source = args.evaluation / 'fixed_policy_traces'
    manifest = json.loads((source / 'manifest.json').read_text())
    if len(manifest['episodes']) != 128 or manifest['provenance']['policy_sha256'] != sha(args.policy):
        raise ValueError('Require every measured first-episode trace')
    entries = sorted(manifest['episodes'], key=lambda x: x['source_training_index'])
    frames, dispatched, rows = [], [], []
    native_rows = {r['source_training_index']: r for r in json.loads((args.evaluation / 'episodes.json').read_text())}
    for entry in entries:
        path = source / entry['data_file']
        if sha(path) != entry['data_sha256']:
            raise ValueError('Recorded trace changed')
        with np.load(path, allow_pickle=False) as z:
            raw = unpack_public(z['packed_public'])
            commands = z['dispatched_actions'].copy()
        if (raw.ndim != 2 or raw.shape[1] != 794 or not 1 <= len(raw) <= 800
                or commands.shape != (len(raw), 18)
                or hashlib.sha256(raw.tobytes()).hexdigest() != entry['full_public794_sha256']):
            raise ValueError('Require intact native observations and commands')
        frames.append(raw)
        dispatched.append(commands)
        rows.append(native_rows[entry['source_training_index']])
    # Some failed episodes genuinely terminate before 8s. Replay only their
    # shared real prefix, without inventing post-termination observations.
    prefix = min(len(frame) for frame in frames)
    frames = np.stack([frame[:prefix] for frame in frames]).transpose(1, 0, 2)
    dispatched = np.stack([command[:prefix] for command in dispatched]).transpose(1, 0, 2)
    policy = torch.load(args.policy, map_location='cpu', weights_only=False)
    production = ROOT / 'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py'
    builder.selected_definitions(production, ['load_shared_factory', 'SharedTargetResidualController'], builder.__dict__)
    controller = builder.SharedTargetResidualController(policy, 128, 'cpu')
    controller.reset()
    default = torch.tensor(manifest['provenance']['default_joint_position_rad'])
    limits = torch.tensor(manifest['provenance']['joint_limits_rad'])
    requested, actual, gates = [], [], []
    max_errors = np.zeros(128)
    gate = PublicAerialProximityGate()
    with torch.inference_mode():
        for tick, frame in enumerate(frames):
            raw = torch.from_numpy(frame.copy())
            result = controller.action(TensorDict({'policy': raw}, batch_size=[128]), default, limits)
            max_errors = np.maximum(max_errors, np.abs(result.numpy() - dispatched[tick]).max(axis=1))
            requested.append(controller.mixer.last_requested.numpy())
            actual.append(result.numpy())
            gates.append(gate(raw).numpy().ravel())
    if not np.isfinite(max_errors).all() or max_errors.max() > 5e-4:
        raise ValueError('CPU replay disagrees with native dispatched commands: ' + str(max_errors.max()))
    lower, span = limits[:, 0].numpy(), (limits[:, 1] - limits[:, 0]).numpy()
    requested = lower + .5 * (np.asarray(requested) + 1) * span
    actual = lower + .5 * (np.asarray(actual) + 1) * span
    measured = default.numpy() + frames[:, :, 9:27]
    gates = np.asarray(gates)
    stages = {s['training_index']: s for s in summary['stages']}
    cases = []
    for lane, row in enumerate(rows):
        clock = frames[:, lane, 775]
        if row['had_contact']:
            end = float(row['contact_time_s'])
            anchor = 'actual_first_racket_contact'
        else:
            eligible = (clock >= 2.2) & (clock <= 3.3) & (frames[:, lane, 67] > .5)
            distance = np.linalg.norm(frames[:, lane, 61:64] - frames[:, lane, 45:48], axis=1)
            end = float(clock[np.argmin(np.where(eligible, distance, np.inf))])
            anchor = 'recorded_closest_ball_tool_in_2.2_to_3.3s'
        if clock[-1] < max(3.3, end):
            raise ValueError('Real shared prefix does not cover the contact window')
        window = (clock >= end - .1) & (clock < end)
        if not window.any():
            raise ValueError('Missing precontact command samples')
        delta = actual[window, lane] - measured[window, lane]
        slew = np.abs(requested[window, lane, 12:] - actual[window, lane, 12:])
        stage = stages[row['source_training_index']]
        group = ('qualified_air' if stage['air'] else
                 'racket_contact_without_strings' if row['had_contact'] and not row['string_hit'] else
                 'no_racket_contact' if not row['had_contact'] else 'other_unqualified_air')
        cases.append(dict(training_index=row['source_training_index'], group=group,
                          window_anchor=anchor, window_end_s=end, samples=int(window.sum()),
                          replay_max_command_error=float(max_errors[lane]),
                          arm_slew_limited_component_fraction=float((slew > 1e-4).mean()),
                          arm_slew_removed_abs_mean_rad=float(slew.mean()),
                          arm_slew_removed_max_rad=float(slew.max()),
                          arm_target_issue_error_rms_rad=float(np.sqrt(np.mean(delta[:, 12:] ** 2))),
                          leg_target_issue_error_rms_rad=float(np.sqrt(np.mean(delta[:, :12] ** 2))),
                          proposed_approach_gate_min=float(gates[window, lane].min()),
                          proposed_approach_gate_mean=float(gates[window, lane].mean())))
    fields = ['arm_slew_limited_component_fraction', 'arm_slew_removed_abs_mean_rad',
              'arm_slew_removed_max_rad', 'arm_target_issue_error_rms_rad',
              'leg_target_issue_error_rms_rad', 'proposed_approach_gate_mean']
    grouped = {}
    for group in sorted({c['group'] for c in cases}):
        members = [c for c in cases if c['group'] == group]
        grouped[group] = dict(count=len(members), median={k: float(np.median([c[k] for c in members])) for k in fields})
    result = dict(schema='native-precontact-command-execution-audit-v2', source_evaluation=str(args.evaluation.resolve()),
                  policy_sha256=sha(args.policy), trace_manifest_sha256=sha(source / 'manifest.json'),
                  production_controller_sha256=sha(production), max_command_replay_error=float(max_errors.max()),
                  actual_shared_prefix_control_steps=prefix,
                  command_tolerance=5e-4, groups=grouped, cases=cases, formal_acceptance=False,
                  scope='CPU replay of actual native command history; 100ms before actual contact or recorded closest approach. '
                        'Target issue error is not a servo response measurement or proof of failure causality. '
                        'The new approach gate is evaluated on old states; no new policy was physically executed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.output)
    print(json.dumps(dict(max_command_replay_error=result['max_command_replay_error'], groups=grouped)))


if __name__ == '__main__':
    main()
