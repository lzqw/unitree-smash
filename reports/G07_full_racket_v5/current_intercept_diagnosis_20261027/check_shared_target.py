"""Run all shared y/z reference proposals on two actual native input streams.

The unchanged proposal must replay the actual GPU stream at the original
5e-4 tolerance. Modified proposals prove bounded commands only, not success.
"""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2].parent/'unitree_smash_contact_recovery'
sys.path.insert(0, str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree
from shared_target_controller import SharedTargetController, candidates


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    factory_path = BASE/'recovery_prior_native_20261027/prior_controller.py'
    spec = importlib.util.spec_from_file_location('shared_bias_preserved_source', factory_path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    proof = read(HERE/'ACTUAL_FORECAST_AND_MISS_CHECK.json')
    assert all(sha(Path(p)) == h for p, h in proof['source_sha256'].items())
    files = [Path(__file__), HERE/'shared_target_controller.py', HERE/'ACTUAL_FORECAST_AND_MISS_CHECK.json', factory_path]
    results = []
    for name, suffix, policy_path in [
        ('early_lateral7_best_cold128_01', '001', BASE/'early_lateral7_n128_01/best_candidate.pt'),
        ('windowed_swing7_n128_01', '003', BASE/'windowed_swing7_n128_01/candidate_003.pt')]:
        run = BASE/name
        inputs_path, data_path = run/'INPUTS.json', run/f'public_and_actions_{suffix}.npz'
        inputs = read(inputs_path)
        defaults, limits = [torch.tensor(inputs[k], dtype=torch.float32) for k in
                            ['default_joint_position_rad', 'joint_limits_rad']]
        source = torch.load(policy_path, map_location='cpu', weights_only=False)
        assert source['native_execution']
        untouched_source = copy.deepcopy(source)
        with np.load(data_path) as data:
            public, native, valid = [data[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
            assert data['training_indices'].tolist() == [0, 29, 69, 100, 112, 123]
        assert public.shape == (800, 6, 794)
        files.extend([policy_path, inputs_path, data_path])
        baseline = factory.make_base(source, 6, 'cpu')
        expected, expected_offsets, expected_parameters = [], [], []
        old_clock = None
        for obs in public:
            if old_clock is not None:
                reset = torch.tensor(np.flatnonzero(obs[:, 775] < old_clock-1e-6))
                if len(reset):
                    baseline.reset(reset)
            expected.append(baseline.action({'policy': torch.from_numpy(obs)}, defaults, limits).clone())
            expected_offsets.append(baseline.reference_controller.target_offset_world.clone())
            expected_parameters.append(baseline.reference_controller.parameters.clone())
            old_clock = obs[:, 775].copy()
        for candidate in candidates():
            controller = SharedTargetController(factory.make_base(source, 6, 'cpu'), 6, 'cpu',
                bias_world_m=candidate['bias_world_m'])
            old_clock = previous_target = None
            ever_active = np.zeros(6, dtype=bool)
            maximum_native_error = maximum_action_change = maximum_arm_rate = 0.
            active_count = changed_command_rows = 0
            digest = hashlib.sha256()
            for tick, (obs, original, keep, base) in enumerate(zip(public, native, valid, expected)):
                reset = np.zeros(6, dtype=bool)
                if old_clock is not None:
                    reset = obs[:, 775] < old_clock-1e-6
                    ids = torch.tensor(np.flatnonzero(reset))
                    if len(ids):
                        controller.reset(ids)
                        ever_active[reset] = False
                values = torch.from_numpy(obs)
                before = values.clone()
                original_offset_object = controller.reference_controller.target_offset_world
                action = controller.action({'policy': values}, defaults, limits)
                assert torch.equal(values, before) and torch.isfinite(action).all() and (abs(action) <= 1).all()
                assert controller.reference_controller.target_offset_world is original_offset_object
                assert torch.equal(controller.reference_controller.target_offset_world, expected_offsets[tick])
                assert torch.equal(controller.reference_controller.parameters, expected_parameters[tick])
                assert torch.equal(action[:, :12], base[:, :12])
                if keep.any():
                    maximum_native_error = max(maximum_native_error, float(abs(action.numpy()[keep]-original[keep]).max()))
                maximum_action_change = max(maximum_action_change, float(abs(action-base).max()))
                changed_command_rows += int((abs(action-base).amax(1) > 0).sum())
                if controller.changed:
                    diagnostic = controller.last_diagnostic
                    active = diagnostic['active'].numpy()
                    assert not np.any(active & ((obs[:, 67] <= .5) | (obs[:, 773] >= .5) | (obs[:, 775] < 2.)))
                    delta = diagnostic['added_world_target_bias_m']
                    assert torch.equal(delta[:, 0], torch.zeros(6))
                    assert (abs(delta) <= abs(controller.bias)[None]+1e-7).all()
                    ever_active |= active
                    active_count += int(active.sum())
                    assert torch.equal(action[~ever_active], base[~ever_active])
                target = limits[:, 0]+.5*(action+1)*(limits[:, 1]-limits[:, 0])
                if old_clock is not None:
                    advancing = (~reset) & (obs[:, 775] > old_clock+1e-6)
                    if advancing.any():
                        dt = torch.from_numpy(obs[advancing, 775]-old_clock[advancing])[:, None]
                        maximum_arm_rate = max(maximum_arm_rate,
                            float((abs(target[advancing, 12:]-previous_target[advancing, 12:])/dt).max()))
                digest.update(action.numpy().tobytes())
                old_clock, previous_target = obs[:, 775].copy(), target.clone()
            assert maximum_arm_rate <= 5.0005
            assert same_tree(source, untouched_source)
            if not controller.changed:
                assert maximum_action_change == 0. and maximum_native_error <= 5e-4
            else:
                assert active_count > 0 and changed_command_rows > 0
            # An actual-input partial reset must preserve every other lane.
            twin = SharedTargetController(factory.make_base(source, 6, 'cpu'), 6, 'cpu',
                bias_world_m=candidate['bias_world_m'])
            controller.reset()
            for obs in public[:250]:
                controller.action({'policy': torch.from_numpy(obs)}, defaults, limits)
                twin.action({'policy': torch.from_numpy(obs)}, defaults, limits)
            controller.reset(torch.tensor([2]))
            hybrid = public[250].copy()
            hybrid[2] = public[0, 2]
            a = controller.action({'policy': torch.from_numpy(hybrid)}, defaults, limits)
            b = twin.action({'policy': torch.from_numpy(public[250])}, defaults, limits)
            other = torch.arange(6) != 2
            assert torch.equal(a[other], b[other]) and torch.equal(controller.valid_plan[other], twin.valid_plan[other])
            item = dict(run=name, candidate=candidate, actual_public_rows=4800,
                valid_first_episode_rows=int(valid.sum()), active_precontact_reference_rows=active_count,
                changed_command_rows=changed_command_rows, maximum_source_action_change=maximum_action_change,
                CPU_vs_actual_GPU_max_action_error=(maximum_native_error if not controller.changed else None),
                maximum_arm_target_rate_rad_s=maximum_arm_rate,
                original_arm_rate_check=5.0005, original_action_tolerance=.0005,
                bounded_finite=True, original_leg_commands_bitwise_preserved=True,
                prefix_before_first_activation_bitwise_preserved=True,
                causal_offsets_and_parameters_restored=True, actors_and_normalizers_unchanged=True,
                partial_reset_preserves_other_lanes=True, command_stream_sha256=digest.hexdigest(),
                native_contact_or_recovery_improvement_proven=False)
            results.append(item)
            print(json.dumps(item), flush=True)
    record = dict(checked_at=datetime.now(timezone.utc).isoformat(), results=results,
        actual_streams=2, candidates_per_stream=5, actual_primary_commands=48000,
        modified_candidates_are_not_source_equivalence_checks=True,
        original_joint_limits_guard_and_arm_rate_retained=True,
        shared_target_bias_changes_reference_not_geometry=True,
        new_native_execution=False, queued_for_GPU=False, formal_cases_read=False,
        formal_acceptance=False, goal_complete=False, github_upload_performed=False,
        source_sha256={str(p): sha(p) for p in sorted(set(files))})
    with (HERE/'CPU_SHARED_TARGET_CHECK.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
