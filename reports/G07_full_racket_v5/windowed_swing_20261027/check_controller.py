"""Exercise bounded windowed policies on4800 actual native public inputs."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'unitree_smash_contact_recovery'
sys.path.insert(0, str(RUNTIME))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('window_controller_check', HERE / 'window_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    origin = ROOT / 'reports/G07_full_racket_v5/origin0_measured_swing_velocity_20261027/RESULT.json'
    proposal = json.loads(origin.read_text())
    source_path = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    data_path = ROOT / 'reports/G07_full_racket_v5/measured_swing9_n128_01/public_and_actions_000.npz'
    inputs_path = data_path.parent / 'INPUTS.json'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    inputs = json.loads(inputs_path.read_text())
    assert sha(source_path) == inputs['source_policy_sha256'] == proposal['sources_sha256'][str(source_path)]
    defaults, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                        ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(data_path) as data:
        public, native, valid = [data[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
        assert data['training_indices'].tolist() == [0, 29, 69, 100, 112, 123]
    assert public.shape == (800, 6, 794) and valid.sum() == 4800
    definitions = [dict(candidate_id=0, name='unchanged_source', joint_velocity_rad_s=None)]
    for row in proposal['candidates'][1:7]:
        assert set(row['overrides']) == {'swing_joint_velocity_rad_s'}
        definitions.append(dict(candidate_id=row['candidate_id'], name='windowed_'+row['name'],
            joint_velocity_rad_s=row['overrides']['swing_joint_velocity_rad_s']))
    results = []
    for candidate in definitions:
        original = module.make_source(source, 6, 'cpu')
        changed = module.WindowedSwingController(source, 6, 'cpu', candidate['joint_velocity_rad_s'])
        previous_clock = previous_target = None
        ever = torch.zeros(6, dtype=torch.bool)
        pre_rows = outside_rows = changed_rows = 0
        maximum_rate = maximum_native_error = maximum_delta = 0.
        maximum_displacement = 0.
        for values, actual_native in zip(public, native):
            obs = torch.from_numpy(values)
            snapshot = obs.clone()
            reset = torch.zeros(6, dtype=torch.bool)
            if previous_clock is not None:
                reset = obs[:, 775] < previous_clock - 1e-5
                ids = torch.where(reset)[0]
                if len(ids):
                    original.reset(ids)
                    changed.reset(ids)
                    ever[ids] = False
            expected = original.action({'policy': obs}, defaults, limits)
            result = changed.action({'policy': obs}, defaults, limits)
            assert torch.equal(obs, snapshot)
            assert result.shape == (6, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
            assert torch.equal(result[:, :12], expected[:, :12])
            diagnostic = changed.last_diagnostic
            assert torch.equal(changed.reference.parameters, original.reference_controller.parameters)
            assert torch.equal(changed.reference.event_time, original.reference_controller.event_time)
            assert torch.equal(changed.reference.target_offset_world, original.reference_controller.target_offset_world)
            active = diagnostic['active'] & (candidate['candidate_id'] != 0)
            ever |= active
            assert torch.equal(result[~ever], expected[~ever])
            pre_rows += int((~ever).sum())
            outside = ~active
            assert torch.equal(changed.teacher.last_arm_targets_before_slew[outside],
                               original.reference_controller.teacher.last_arm_targets_before_slew[outside])
            outside_rows += int(outside.sum())
            changed_rows += int((result != expected).any(1).sum())
            maximum_displacement = max(maximum_displacement, float(abs(diagnostic['feedforward_joint_displacement_rad']).max()))
            maximum_delta = max(maximum_delta, float(abs(result-expected).max()))
            maximum_native_error = max(maximum_native_error, float(abs(result.numpy()-actual_native).max()))
            targets = limits[:, 0] + .5*(result+1)*(limits[:, 1]-limits[:, 0])
            if previous_target is not None:
                good = ~reset
                if good.any():
                    maximum_rate = max(maximum_rate, float(abs(targets[good, 12:]-previous_target[good, 12:]).max()) / .01)
            previous_target = targets.clone()
            previous_clock = obs[:, 775].clone()
        assert maximum_rate <= 5.0005
        assert same_tree(source, before)
        assert same_tree(changed.logs, original.logs)
        # Joint trajectory derivative at its unchanged center is the declared
        # new seed; this checks feedforward geometry, not physical tracking.
        if candidate['candidate_id']:
            assert changed_rows > 0 and maximum_delta > 0.
            old = diagnostic['original_velocity'].double()
            target = changed.velocity.double()[None].expand_as(old)
            width = diagnostic['half_window_s'].double()
            step = torch.full_like(width, 1e-6)
            derivative = old + module.window_envelope(step, width)[:, None]*(target-old)
            assert float(abs(derivative-target).max()) < 1e-8
            assert torch.equal(module.window_envelope(torch.zeros_like(width), width), torch.ones_like(width))
            assert torch.equal(module.window_envelope(width, width), torch.zeros_like(width))
        else:
            assert maximum_delta == 0. and maximum_native_error <= 5e-4
        # A reset of one lane cannot alter any other lane's next command.
        reset_control = module.WindowedSwingController(source, 6, 'cpu', candidate['joint_velocity_rad_s'])
        reset_control.action({'policy': torch.from_numpy(public[0])}, defaults, limits)
        changed.reset()
        changed.action({'policy': torch.from_numpy(public[0])}, defaults, limits)
        changed.reset(torch.tensor([0]))
        reset_values = public[1].copy()
        reset_values[0] = public[0, 0]
        a = changed.action({'policy': torch.from_numpy(reset_values)}, defaults, limits)
        b = reset_control.action({'policy': torch.from_numpy(public[1])}, defaults, limits)
        assert torch.equal(a[1:], b[1:])
        results.append(dict(candidate_id=candidate['candidate_id'], name=candidate['name'],
            actual_public_rows=int(valid.sum()), pre_first_window_commands_bitwise_equal_rows=pre_rows,
            outside_window_original_guarded_teacher_targets_bitwise_equal_rows=outside_rows,
            changed_command_rows=changed_rows, maximum_normalized_action_change=maximum_delta,
            maximum_CPU_vs_original_GPU_action_difference=maximum_native_error,
            maximum_feedforward_joint_displacement_rad=maximum_displacement,
            maximum_arm_command_target_rate_rad_s=maximum_rate,
            source_parameters_forecasts_and_logs_unchanged=True, leg_commands_bitwise_unchanged=True,
            partial_reset_preserves_other_lanes=True))
    hashes = {str(p): sha(p) for p in [source_path, data_path, inputs_path, origin,
        HERE / 'window_controller.py', Path(__file__),
        HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py',
        HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']}
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidates=definitions, results=results,
        actual_public_rows_per_candidate=int(valid.sum()), source_sha256=hashes,
        checkpoint_tree_unchanged=True, source_actor_and_normalizer_unchanged=True,
        original_arm_target_rate_limit_rad_s=5., floating_point_rate_check_tolerance_rad_s=.0005,
        original_command_CPU_GPU_tolerance=5e-4,
        new_physical_tracking_or_contacts_or_recovery_executed=False,
        native_execution=False, formal_acceptance=False)
    with (HERE / 'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(candidate_count=len(results), actual_public_rows_per_candidate=int(valid.sum()),
        source_action_error=results[0]['maximum_CPU_vs_original_GPU_action_difference'],
        maximum_arm_rate=max(r['maximum_arm_command_target_rate_rad_s'] for r in results))))


if __name__ == '__main__':
    main()
