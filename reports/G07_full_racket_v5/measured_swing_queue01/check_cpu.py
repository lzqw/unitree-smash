"""Replay all nine proposals using actual794-channel public commands and v5 code."""
import ast
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

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
PROPOSALS = MAIN / 'reports/G07_full_racket_v5/origin0_measured_swing_velocity_20261027'
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.rl.contact_recovery_source_v5 import verify_recovery_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def main():
    for path in [HERE / 'evaluate_swing.py', HERE / 'run.py', Path(__file__)]:
        ast.parse(path.read_text(), filename=str(path))
    source_path = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    protocol = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    source_check = verify_recovery_source(source_path, source,
        MAIN / 'reports/G07_full_racket_v5/teacher_contact_recovery_cold128_02', ROOT, sha(protocol))
    spec = importlib.util.spec_from_file_location('measured_swing_cpu', PROPOSALS / 'swing_controller.py')
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    path = MAIN / 'reports/G07_full_racket_v4/gain_body8_best_fixed_train128_01/public_and_actions_first2.npz'
    metadata_path = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/INPUTS.json'
    metadata = json.loads(metadata_path.read_text())
    default, limits = [torch.tensor(metadata[name], dtype=torch.float32) for name in
        ['default_joint_position_rad', 'joint_limits_rad']]
    proposal = json.loads((PROPOSALS / 'RESULT.json').read_text())
    with np.load(path) as data:
        public, native, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    assert public.shape == (800, 2, 794) and int(valid.sum()) == 1600
    baseline = make_rally_controller(source, 2, 'cpu')
    results = []
    with torch.inference_mode():
        expected = []
        previous = None
        for value in public:
            obs = torch.from_numpy(value)
            if previous is not None:
                ids = torch.where(obs[:, 775] < previous - 1e-5)[0]
                if len(ids):
                    baseline.reset(ids)
            expected.append(baseline.action({'policy': obs}, default, limits).clone())
            previous = obs[:, 775].clone()
        for candidate in proposal['candidates']:
            controller = factory.MeasuredSwingController(source, 2, 'cpu', candidate['overrides'])
            previous = previous_target = None
            maximum_native = maximum_source = maximum_slew = 0.
            for value, actual, keep, reference in zip(public, native, valid, expected):
                obs = torch.from_numpy(value)
                reset = torch.zeros(2, dtype=torch.bool)
                if previous is not None:
                    reset = obs[:, 775] < previous - 1e-5
                    ids = torch.where(reset)[0]
                    if len(ids):
                        controller.reset(ids)
                result = controller.action({'policy': obs}, default, limits)
                assert result.shape == (2, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
                maximum_source = max(maximum_source, float(abs(result - reference).max()))
                maximum_native = max(maximum_native, float(abs(result.numpy()[keep] - actual[keep]).max()))
                target = limits[:, 0] + .5 * (result + 1) * (limits[:, 1] - limits[:, 0])
                if previous_target is not None and (~reset).any():
                    maximum_slew = max(maximum_slew, float(abs(target[~reset, 12:] - previous_target[~reset, 12:]).max()) / .01)
                previous_target, previous = target.clone(), obs[:, 775].clone()
            assert maximum_slew <= 5.0005
            if candidate['candidate_id'] == 0:
                assert maximum_source == 0 and maximum_native <= 5e-4
            else:
                assert maximum_source > 0
            other = controller.reference.parameters[1].clone()
            controller.reset(torch.tensor([0]))
            assert torch.equal(controller.reference.parameters[1], other)
            assert same(source, before)
            results.append(dict(candidate_id=candidate['candidate_id'], actual_public_rows=int(valid.sum()),
                maximum_source_action_difference=maximum_source, maximum_actual_GPU_action_difference=maximum_native,
                maximum_arm_target_rate_rad_s=maximum_slew, partial_reset_other_lane_unchanged=True))
    assert sha(source_path) == source_check['source_policy_sha256']
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), all_nine_candidates_checked=True,
        source_check=source_check, results=results, full_source_tensor_tree_unchanged=True,
        actual_runtime_worktree=str(ROOT), actual_control_module=sys.modules[make_rally_controller.__module__].__file__,
        public794_channel_inputs_only=True, original_action_tolerance=5e-4,
        target_rate_limit_rad_s=5., float_rate_check_tolerance_rad_s=.0005,
        program_sha256={str(p): sha(p) for p in [Path(__file__), HERE / 'evaluate_swing.py', HERE / 'run.py', PROPOSALS / 'swing_controller.py']},
        public_arrays_sha256=sha(path), native_execution=False, formal_acceptance=False)
    with (HERE / 'PRE_NATIVE_CPU_CHECK.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(all_nine_candidates_checked=True, rows_per_candidate=1600,
        unchanged_candidate_error=results[0]['maximum_actual_GPU_action_difference'],
        max_arm_target_rate=max(row['maximum_arm_target_rate_rad_s'] for row in results))))


if __name__ == '__main__':
    main()
