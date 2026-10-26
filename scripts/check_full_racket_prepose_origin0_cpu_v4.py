#!/usr/bin/env python3
"""Replay real public inputs to check reference identity and preparation guards."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_prepose_branch_v4 import (
    PreposeBranchControllerV4, GroupedPreposeControllerV4, validate_overrides)
from quadsmash.control.full_racket_contact_aim_v4 import make_rally_controller
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
    original = make_rally_controller(source, 2, 'cpu')
    candidates = proposal['candidates']
    controllers = [PreposeBranchControllerV4(source, candidate['overrides'], 2, 'cpu') for candidate in candidates]
    grouped = GroupedPreposeControllerV4(source, candidates, 2, 'cpu')
    maximum_reference = maximum_native = maximum_grouped = maximum_slew = 0.
    changed = np.zeros(len(candidates), dtype=int)
    previous = None
    # The fixed evaluator does not call controller.reset after done. Clock
    # restarts inside each existing limiter still apply to later invalid rows.
    with torch.inference_mode():
        for observation, actual, keep in zip(observations, actions, valid):
            public = torch.from_numpy(observation)
            reference = original.action({'policy': public}, default, limits)
            batch = grouped.action({'policy': public.repeat(len(candidates), 1)}, default, limits)
            for i, controller in enumerate(controllers):
                rate = controller.reference.teacher.arm_target_rate
                old_target, old_clock, initialized = rate.target.clone(), rate.clock.clone(), rate.initialized.clone()
                command = controller.action({'policy': public}, default, limits)
                assert torch.isfinite(command).all() and (abs(command) <= 1).all() and command.shape == (2, 18)
                maximum_grouped = max(maximum_grouped, float(abs(command-batch[2*i:2*i+2]).max()))
                if keep.any():
                    if i == 0:
                        maximum_reference = max(maximum_reference, float(abs(command.numpy()[keep]-reference.numpy()[keep]).max()))
                        maximum_native = max(maximum_native, float(abs(command.numpy()[keep]-actual[keep]).max()))
                    changed[i] += int((abs(command.numpy()[keep]-reference.numpy()[keep]).max(axis=1) > 1e-6).sum())
                restarted = (~initialized) | (public[:, 775] < old_clock-1e-6)
                observed = (default[12:][None]+public[:, 21:27]).clamp(limits[12:, 0], limits[12:, 1])
                prior = torch.where(restarted[:, None], observed, old_target)
                dt = torch.where(restarted, torch.full_like(old_clock, .01), (public[:, 775]-old_clock).clamp(0., .01))
                excess = (abs(rate.target-prior)-5.*dt[:, None]).max().item()
                maximum_slew = max(maximum_slew, excess)
                assert excess <= 2e-7 and rate.rate == 5.
            previous = public[:, 775].clone()
    assert maximum_reference == 0. and maximum_native <= 5e-4 and maximum_grouped == 0.
    assert all(n > 0 for n in changed[1:]) and changed[0] == 0
    # Exercise partial reset after causal planning: runtime overrides must be
    # reinstalled and an unrelated lane's planned correction must survive.
    for controller in controllers[1:]:
        reference = controller.reference
        untouched = reference.parameters[1].clone()
        controller.reset(torch.tensor([0]))
        assert torch.equal(reference.parameters[0], controller.runtime_parameters[0])
        assert torch.equal(reference.parameters[1], untouched) and not reference.planned[0]
        assert reference.teacher.joint_preparation is not None and reference.teacher.face_feedback is not None
    grouped.reset(torch.tensor([0, 5, 10]))
    for lane in [0, 5, 10]:
        controller = grouped.controllers[lane//2]
        assert not controller.reference.planned[lane % 2]
        assert torch.equal(controller.reference.parameters[lane % 2], controller.runtime_parameters[lane % 2])
    for invalid in [{'hit_joint_position_rad': [9.]*6}, {'swing_joint_velocity_rad_s': [5.01]*6},
                    {'walk_waypoint_delta_m': .21}, {'desired_normal_world': [2., 0., 0.]},
                    {'disable_physical_guard': True}, {'nominal_projected_pitch_rad': float('nan')}]:
        try:
            validate_overrides(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid override accepted')
    assert same_tree(source, before), 'Source checkpoint changed in memory'
    assert sha(source_path) == proposal['source_policy_sha256']
    paths = [ROOT/'quadsmash/control/full_racket_prepose_branch_v4.py',
             ROOT/'scripts/prepare_full_racket_prepose_origin0_v4.py', Path(__file__)]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        proposal_sha256=sha(proposals), source_policy_sha256=sha(source_path),
        actual_valid_source_public_rows=int(valid.sum()), exact_reference_max_action_error=maximum_reference,
        CPU_vs_recorded_source_GPU_max_action_error=maximum_native, original_action_tolerance=5e-4,
        grouped_vs_separate_max_action_error=maximum_grouped, maximum_slew_excess_rad=maximum_slew,
        changed_public_rows_by_candidate=changed.tolist(),
        source_checkpoint_tree_unchanged=True, partial_reset_and_group_lane_mapping_checked=True,
        invalid_override_rejections_checked=6, source_sha256={str(path):sha(path) for path in paths},
        initial_cpu_attempt_failure='Isolated worktree lacked the USD asset. Copied original11508-byte asset with pinned SHA; no main source was changed.',
        scope='Actual public-input CPU command replay and software guards only; no new physical policy execution or success measurement.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    output.parent.mkdir(exist_ok=True, parents=True)
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
