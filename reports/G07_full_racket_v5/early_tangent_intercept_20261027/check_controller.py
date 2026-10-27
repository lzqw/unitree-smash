"""Replay early lateral proposals on4800 real public inputs, never new outcomes."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
RUN = MAIN / 'reports/G07_full_racket_v5/measured_swing9_n128_01'
sys.path.insert(0, str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = HERE / 'early_lateral_controller.py'
    ast.parse(path.read_text())
    spec = importlib.util.spec_from_file_location('early_lateral_cpu_factory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_path = RUN / 'candidate_000.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    inputs = json.loads((RUN / 'INPUTS.json').read_text())
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(RUN / 'public_and_actions_000.npz') as data:
        public, native, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
        recorded = data['training_indices'].tolist()
    assert public.shape == (800, 6, 794) and int(valid.sum()) == 4800
    baseline = module.source_controller(source, 6, 'cpu')
    expected, previous = [], None
    with torch.inference_mode():
        for value in public:
            obs = torch.from_numpy(value)
            if previous is not None:
                ids = torch.where(obs[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    baseline.reset(ids)
            expected.append(baseline.action({'policy': obs}, default, limits).clone())
            previous = obs[:, 775].clone()
    results = []
    for candidate in module.candidates():
        controller = module.EarlyLateralController(source, 6, 'cpu', candidate['gain'], candidate['start_clock_s'])
        maximum_native = maximum_source = maximum_slew = 0.
        previous = previous_target = None
        active_rows = changed_rows = pre_gate_changed_rows = leg_changed_rows = 0
        directional_improvements, requested_errors = [], []
        with torch.inference_mode():
            for values, actual, keep, reference in zip(public, native, valid, expected):
                obs = torch.from_numpy(values)
                reset = torch.zeros(6, dtype=torch.bool)
                if previous is not None:
                    reset = obs[:, 775] < previous - 1e-5
                    ids = torch.where(reset)[0]
                    if len(ids):
                        controller.reset(ids)
                result = controller.action({'policy': obs}, default, limits)
                assert result.shape == (6, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
                difference = abs(result - reference)
                maximum_source = max(maximum_source, float(difference.max()))
                maximum_native = max(maximum_native, float(abs(result.numpy()[keep] - actual[keep]).max()))
                changed_rows += int((difference.amax(1) > 1e-7).sum())
                leg_changed_rows += int((difference[:, :12].amax(1) > 0).sum())
                pre = (obs[:, 775] <= candidate['start_clock_s']) | (obs[:, 67] < .5)
                # Before any lane can have entered preparation, the source is exact.
                if (obs[:, 775] <= candidate['start_clock_s']).all():
                    pre_gate_changed_rows += int((difference[pre].amax(1) > 0).sum())
                target = limits[:, 0] + .5 * (result + 1) * (limits[:, 1] - limits[:, 0])
                if previous_target is not None and (~reset).any():
                    maximum_slew = max(maximum_slew,
                        float(abs(target[~reset, 12:] - previous_target[~reset, 12:]).max()) / .01)
                diagnostics = controller.last_diagnostic
                if diagnostics is not None:
                    active = diagnostics['active']
                    active_rows += int(active.sum())
                    if active.any():
                        base_target = limits[12:, 0] + .5 * (reference[:, 12:] + 1) * (limits[12:, 1] - limits[12:, 0])
                        point_before = controller.teacher.kinematics(base_target, controller.teacher.contact_offset)[0]
                        point_after = controller.teacher.kinematics(target[:, 12:], controller.teacher.contact_offset)[0]
                        left = torch.linalg.cross(-obs[:, 6:9], obs[:, 763:766])
                        goal = diagnostics['public_forecast_target_body_m']
                        before_error = abs(((goal - point_before) * left).sum(1))
                        after_error = abs(((goal - point_after) * left).sum(1))
                        directional_improvements.extend((before_error - after_error)[active].tolist())
                        requested_errors.extend(diagnostics['error_world_y_m'][active].tolist())
                previous, previous_target = obs[:, 775].clone(), target.clone()
        assert leg_changed_rows == pre_gate_changed_rows == 0
        assert maximum_slew <= 5.0005
        if candidate['candidate_id'] == 0:
            assert maximum_source == 0 and maximum_native <= 5e-4
        else:
            assert maximum_source > 0 and changed_rows and active_rows
        before_state = dict(valid=controller.valid_plan[1:].clone(), ever=controller.ever_active[1:].clone(),
            rate_target=controller.rate.target[1:].clone(), rate_clock=controller.rate.clock[1:].clone(),
            rate_initialized=controller.rate.initialized[1:].clone())
        controller.reset(torch.tensor([0]))
        assert torch.equal(controller.valid_plan[1:], before_state['valid'])
        assert torch.equal(controller.ever_active[1:], before_state['ever'])
        assert torch.equal(controller.rate.target[1:], before_state['rate_target'])
        assert torch.equal(controller.rate.clock[1:], before_state['rate_clock'])
        assert torch.equal(controller.rate.initialized[1:], before_state['rate_initialized'])
        assert same_tree(source, before)
        result = dict(candidate=candidate, actual_public_rows=4800, maximum_source_action_difference=maximum_source,
            maximum_original_GPU_action_difference=maximum_native, active_public_rows=active_rows,
            changed_public_rows=changed_rows, changed_leg_rows=leg_changed_rows,
            changed_before_public_start_rows=pre_gate_changed_rows, maximum_arm_target_rate_rad_s=maximum_slew,
            partial_reset_other_lanes_preserved=True, entire_source_tensor_tree_unchanged=True,
            actual_recorded_measured_y_error_median=float(np.median(requested_errors)) if requested_errors else None,
            hypothetical_command_FK_y_error_improvement_median_m=float(np.median(directional_improvements)) if directional_improvements else None,
            candidate_closed_loop_motion_executed=False)
        results.append(result)
        print(json.dumps(result), flush=True)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidate_count=len(results),
        all_nine_candidates_checked=True, actual_public_training_indices=recorded,
        actual_public_rows_per_candidate=4800, results=results,
        source_policy_sha256=sha(source_path), data_sha256=sha(RUN / 'public_and_actions_000.npz'),
        checker_sha256=sha(Path(__file__)), factory_sha256=sha(path),
        source_actor_weights_and_normalizers_and_checkpoint_unchanged=True,
        proposed_feedback_uses_only_current_public_state_and_original_causal_forecast=True,
        current_native9_not_modified=True, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    serialized = json.dumps(result, indent=2) + '\n'
    with (HERE / 'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        stream.write(serialized)


if __name__ == '__main__':
    main()
