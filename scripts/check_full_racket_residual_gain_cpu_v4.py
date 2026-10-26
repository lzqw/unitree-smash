#!/usr/bin/env python3
"""Exercise gain adaptation on complete real public-input recordings."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_prepose_branch_v4 import make_rally_controller as make_source
from quadsmash.control.full_racket_residual_gain_v4 import ResidualGainControllerV4, validate_gains
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(proposals, output, provenance):
    assert not output.exists()
    proposal = json.loads(proposals.read_text())
    source_path, evaluation = Path(proposal['source_policy']), Path(proposal['source_evaluation'])
    assert sha(source_path) == proposal['source_policy_sha256']
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    metadata = json.loads(provenance.read_text())['provenance']
    default, limits = [torch.tensor(metadata[key], dtype=torch.float32) for key in
                       ('default_joint_position_rad', 'joint_limits_rad')]
    with np.load(evaluation/'public_and_actions_first2.npz') as arrays:
        observations, actions, valid = [arrays[key].copy() for key in ('observations', 'actions', 'valid_first_episode')]
    controllers = [ResidualGainControllerV4(source, candidate['gains'], 2, 'cpu')
                   for candidate in proposal['candidates']]
    original = make_source(source, 2, 'cpu')
    analytic = make_source(source, 2, 'cpu').reference_controller
    changed = np.zeros(8, dtype=int)
    maximum_reference = maximum_native = maximum_zero = maximum_slew = 0.
    previous_clock = None
    last_targets = [None]*8
    # Follow the actual fixed evaluator: no controller.reset call after done.
    # Existing controller clock-restart handling remains responsible for resets.
    with torch.inference_mode():
        for observation, actual, keep in zip(observations, actions, valid):
            public = torch.from_numpy(observation)
            reference = original.action({'policy': public}, default, limits)
            analytic_command = analytic.action({'policy': public}, default, limits)
            for i, controller in enumerate(controllers):
                command = controller.action({'policy': public}, default, limits)
                assert command.shape == (2, 18) and torch.isfinite(command).all() and (abs(command) <= 1).all()
                if keep.any():
                    if i == 0:
                        maximum_reference = max(maximum_reference, float(abs(command-reference).max()))
                        maximum_native = max(maximum_native, float(abs(command.numpy()[keep]-actual[keep]).max()))
                    if i == 6:
                        maximum_zero = max(maximum_zero, float(abs(command-analytic_command).max()))
                    changed[i] += int((abs(command.numpy()[keep]-reference.numpy()[keep]).max(axis=1) > 1e-6).sum())
                targets = limits[:, 0]+.5*(command+1)*(limits[:, 1]-limits[:, 0])
                if previous_clock is not None:
                    continuous = (public[:, 775] >= previous_clock) & keep
                    if continuous.any():
                        dt = (public[:, 775]-previous_clock).clamp(0., .01)
                        excess = (abs(targets[:, 12:]-last_targets[i][:, 12:])-5.*dt[:, None])[continuous].max().item()
                        maximum_slew = max(maximum_slew, excess)
                last_targets[i] = targets
            previous_clock = public[:, 775].clone()
    assert maximum_reference == maximum_zero == 0. and maximum_native <= 5e-4
    assert maximum_slew <= 2e-6, 'Normalized-action roundtrip must preserve original5rad/s targets'
    assert changed[0] == 0 and all(value > 0 for value in changed[1:])
    for controller in controllers:
        controller.reset(torch.tensor([0]))
        for layer in controller.layers:
            assert not layer.mixer.arm_rate.initialized[0]
            assert layer.mixer.arm_rate.initialized[1]
        assert controller.reference.teacher.arm_target_rate.rate == 5.
        assert controller.reference.teacher.joint_preparation is not None
        assert controller.reference.teacher.face_feedback is not None
    rejected = 0
    for value in [[[1., 1.]], [[1., 1.], [1.01, 0.]], [[-1., 0.], [0., 0.]],
                  [[float('nan'), 0.], [0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]:
        try:
            validate_gains(value)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid gain accepted')
    assert same_tree(source, before) and sha(source_path) == proposal['source_policy_sha256']
    paths = [ROOT/'quadsmash/control/full_racket_residual_gain_v4.py',
             ROOT/'scripts/prepare_full_racket_residual_gain_v4.py', Path(__file__)]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        proposal_sha256=sha(proposals), source_policy_sha256=sha(source_path),
        actual_valid_source_public_rows=int(valid.sum()), exact_reference_max_action_error=maximum_reference,
        zero_layers_vs_analytic_max_action_error=maximum_zero,
        CPU_vs_recorded_source_GPU_max_action_error=maximum_native, original_action_tolerance=5e-4,
        maximum_final_command_slew_excess_rad=maximum_slew, normalized_roundtrip_tolerance_rad=2e-6,
        changed_public_rows_by_candidate=changed.tolist(), source_checkpoint_tree_unchanged=True,
        exact_two_layer_order_checked=True, actual_saturation_then_gain_checked=True,
        partial_reset_checked=True, invalid_gain_rejections=rejected,
        source_sha256={str(path):sha(path) for path in paths},
        scope='Full actual public-input replay and software command checks; no new physical performance inferred.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    args = parser.parse_args()
    check(args.proposals, args.output, args.provenance)
