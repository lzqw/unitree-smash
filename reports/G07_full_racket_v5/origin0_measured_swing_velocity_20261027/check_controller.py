"""Verify proposal commands on real public inputs; no new native outcome."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def main():
    spec = importlib.util.spec_from_file_location('recorded_swing_controller', HERE / 'swing_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    data_path = ROOT / 'reports/G07_full_racket_v4/gain_body8_best_fixed_train128_01/public_and_actions_first2.npz'
    metadata_path = ROOT / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/INPUTS.json'
    source = torch.load(path, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    metadata = json.loads(metadata_path.read_text())
    default, limits = [torch.tensor(metadata[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    proposal = json.loads((HERE / 'RESULT.json').read_text())
    assert proposal['sources_sha256'][str(path)] == sha(path)
    with np.load(data_path) as data:
        public, actions, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    results = []
    for candidate in proposal['candidates']:
        controller = module.MeasuredSwingController(source, 2, 'cpu', candidate['overrides'])
        original = make_rally_controller(source, 2, 'cpu')
        previous_clock, previous_target = None, None
        max_delta = max_native = max_slew = 0.
        with torch.inference_mode():
            for values, native, keep in zip(public, actions, valid):
                observation = torch.from_numpy(values)
                reset = torch.zeros(2, dtype=torch.bool)
                if previous_clock is not None:
                    reset = observation[:, 775] < previous_clock - 1e-5
                    ids = torch.where(reset)[0]
                    if len(ids):
                        controller.reset(ids)
                        original.reset(ids)
                actual = controller.action({'policy': observation}, default, limits)
                reference = original.action({'policy': observation}, default, limits)
                assert actual.shape == (2, 18) and torch.isfinite(actual).all()
                assert (abs(actual) <= 1.).all()
                targets = limits[:, 0] + .5 * (actual + 1) * (limits[:, 1] - limits[:, 0])
                max_delta = max(max_delta, float(abs(actual - reference).max()))
                if keep.any():
                    max_native = max(max_native, float(abs(actual.numpy()[keep] - native[keep]).max()))
                if previous_target is not None:
                    checked = torch.where(~reset)[0]
                    if len(checked):
                        max_slew = max(max_slew, float(abs(targets[checked, 12:] - previous_target[checked, 12:]).max()) / .01)
                previous_target = targets.clone()
                previous_clock = observation[:, 775].clone()
        assert max_slew <= 5.0005, (candidate['name'], max_slew)
        if candidate['candidate_id'] == 0:
            assert max_delta == 0. and max_native <= 5e-4
        else:
            assert max_delta > 0.
            expected = torch.tensor(candidate['overrides']['swing_joint_velocity_rad_s'])
            assert torch.equal(controller.reference.parameters[0, 26:32], expected)
        other = controller.reference.parameters[1].clone()
        controller.reset(torch.tensor([0]))
        assert torch.equal(controller.reference.parameters[1], other)
        assert same(source, before)
        results.append(dict(candidate_id=candidate['candidate_id'], name=candidate['name'],
            actual_public_rows=int(valid.sum()), maximum_change_from_source_normalized_action=max_delta,
            maximum_difference_from_old_native_actions=max_native,
            maximum_arm_command_target_rate_rad_s=max_slew,
            partial_reset_preserves_other_lane=True, checkpoint_tree_unchanged=True))
    assert sha(path) == proposal['sources_sha256'][str(path)]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidate_count=len(results), results=results,
        actual_public_rows_per_candidate=int(valid.sum()),
        original_arm_target_rate_limit_rad_s=5., floating_point_rate_check_tolerance_rad_s=.0005,
        unchanged_source_candidate_action_error=results[0]['maximum_change_from_source_normalized_action'],
        source_checkpoint_sha256=sha(path), public_arrays_sha256=sha(data_path), command_metadata_sha256=sha(metadata_path),
        factory_source_sha256=sha(HERE / 'swing_controller.py'), checker_source_sha256=sha(Path(__file__)),
        source_actor_and_normalizer_and_checkpoint_tree_unchanged=True,
        original_source_unchanged_candidate_matches_actual_GPU=True,
        other_candidates_replayed_on_original_policy_inputs_only=True,
        new_joint_tracking_or_collisions_or_recovery_executed=False, native_execution=False, formal_acceptance=False)
    with (HERE / 'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(candidate_count=len(results), actual_public_rows_per_candidate=int(valid.sum()),
                         unchanged_candidate_error=results[0]['maximum_change_from_source_normalized_action'],
                         maximum_arm_target_rate_rad_s=max(row['maximum_arm_command_target_rate_rad_s'] for row in results))))


if __name__ == '__main__':
    main()
