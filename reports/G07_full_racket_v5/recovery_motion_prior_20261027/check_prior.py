"""Check the actual recovered trace and shared prior on real public inputs."""
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
REPORTS = MAIN / 'reports/G07_full_racket_v5'
TRACE = REPORTS / 'residual_contact_recovery_n128_updates15_02/trajectory_traces/lane0069_episode000000.npz'
POLICY = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = HERE / 'prior_controller.py'
    spec = importlib.util.spec_from_file_location('native_recovery_prior_factory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed, tables = module.load_seed()
    source = torch.load(POLICY, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    inputs = json.loads((REPORTS / 'measured_swing9_n128_01/INPUTS.json').read_text())
    default, limits = [torch.tensor(inputs[name], dtype=torch.float32) for name in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    recovered = module.RecoveryPriorController(source, seed, tables, 1, 'cpu',
        group='full', table='raw', alignment='absolute', gain=1.)
    with torch.inference_mode():
        sampled = recovered.sample(torch.from_numpy(tables['clock_s']))
    assert np.array_equal(sampled.numpy(), tables['residual_raw'])
    with np.load(TRACE) as data:
        public = unpack_public(data['packed_public'])
        residuals, native = data['residual_actions'].copy(), data['native_actions'].copy()
    maximum_recovered = 0.
    with torch.inference_mode():
        for values, request, actual in zip(public, residuals, native):
            result = recovered.action({'policy': torch.from_numpy(values[None])}, default, limits)
            assert np.array_equal(recovered.last_request.numpy()[0], request)
            maximum_recovered = max(maximum_recovered, float(abs(result.numpy()[0] - actual).max()))
    assert maximum_recovered <= 5e-4
    measured = REPORTS / 'measured_swing9_n128_01'
    with np.load(measured / 'public_and_actions_000.npz') as data:
        observations, actions, valid = [data[name].copy() for name in ['observations', 'actions', 'valid_first_episode']]
    baseline = module.make_base(source, 6, 'cpu')
    expected = []
    with torch.inference_mode():
        for values in observations:
            expected.append(baseline.action({'policy': torch.from_numpy(values)}, default, limits).clone())
    results = []
    for candidate in module.candidates():
        controller = module.RecoveryPriorController(source, seed, tables, 6, 'cpu',
            **{key: candidate[key] for key in ['group', 'table', 'alignment', 'gain']})
        previous = None
        max_native = max_source = max_slew = max_leg_change = max_arm_change = 0.
        previously_seen = torch.zeros(6, dtype=torch.bool)
        kept_group_differences = before_observed_hit_differences = post_landing_requested_rows = 0
        with torch.inference_mode():
            for values, native_action, keep, base in zip(observations, actions, valid, expected):
                obs = torch.from_numpy(values)
                result = controller.action({'policy': obs}, default, limits)
                assert result.shape == (6, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
                difference = abs(result - base)
                max_source = max(max_source, float(difference.max()))
                max_native = max(max_native, float(abs(result.numpy()[keep] - native_action[keep]).max()))
                if candidate['group'] == 'legs':
                    kept_group_differences += int((difference[:, 12:] > 0).sum())
                elif candidate['group'] == 'arms':
                    kept_group_differences += int((difference[:, :12] > 0).sum())
                if candidate['alignment'] == 'observed_hit':
                    previously_seen |= obs[:, 773] > .5
                    if (~previously_seen).any():
                        before_observed_hit_differences += int((difference[~previously_seen] > 0).sum())
                    grounded_after_hit = previously_seen & (obs[:, 774] > .5)
                    if controller.last_request is not None:
                        post_landing_requested_rows += int((grounded_after_hit &
                            (controller.last_request.abs().amax(1) > 0)).sum())
                target = limits[:, 0] + .5 * (result + 1) * (limits[:, 1] - limits[:, 0])
                base_target = limits[:, 0] + .5 * (base + 1) * (limits[:, 1] - limits[:, 0])
                max_leg_change = max(max_leg_change, float(abs(target[:, :12] - base_target[:, :12]).max()))
                max_arm_change = max(max_arm_change, float(abs(target[:, 12:] - base_target[:, 12:]).max()))
                if previous is not None:
                    max_slew = max(max_slew, float(abs(target[:, 12:] - previous[:, 12:]).max()) / .01)
                previous = target.clone()
        assert max_slew <= 5.0005 and kept_group_differences == before_observed_hit_differences == 0
        if candidate['candidate_id'] == 0:
            assert max_source == 0. and max_native <= 5e-4
        else:
            assert max_source > 0.
        if candidate['alignment'] == 'observed_hit':
            assert post_landing_requested_rows > 0
        other_seen, other_clock = controller.air_hit_seen[1:].clone(), controller.first_public_hit_clock[1:].clone()
        controller.reset(torch.tensor([0]))
        assert torch.equal(controller.air_hit_seen[1:], other_seen)
        assert torch.equal(controller.first_public_hit_clock[1:], other_clock)
        assert same_tree(source, before)
        row = dict(candidate=candidate, actual_public_rows=int(valid.sum()),
            maximum_source_action_difference=max_source, maximum_original_GPU_action_difference=max_native,
            maximum_arm_target_rate_rad_s=max_slew, maximum_leg_target_change_rad=max_leg_change,
            maximum_arm_target_change_rad=max_arm_change, unaffected_group_changed_values=kept_group_differences,
            action_changed_before_observed_air_hit_values=before_observed_hit_differences,
            nonzero_prior_grounded_rows_after_observed_air_hit=post_landing_requested_rows,
            selective_reset_preserves_other_lanes=True, source_actor_and_checkpoint_tree_unchanged=True,
            new_closed_loop_recovery_or_attack_outcome_proven=False)
        results.append(row)
        print(json.dumps(row), flush=True)
    # Also instantiate and replay the source family already exported by the
    # current native search. This does not select or inherit its performance.
    measured_path = measured / 'candidate_000.pt'
    measured_source = torch.load(measured_path, map_location='cpu', weights_only=False)
    measured_controller = module.RecoveryPriorController(measured_source, seed, tables, 6, 'cpu',
        group='full', table='raw', alignment='absolute', gain=0.)
    measured_error = 0.
    with torch.inference_mode():
        for values, native_action, keep in zip(observations, actions, valid):
            result = measured_controller.action({'policy': torch.from_numpy(values)}, default, limits).numpy()
            measured_error = max(measured_error, float(abs(result[keep] - native_action[keep]).max()))
    assert measured_error <= 5e-4
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidates_checked=len(results), results=results,
        exact_table_requests_match_all800_actual_recorded_requests_bitwise=True,
        recovered_trace_actual_public_rows=800, CPU_complete_recovered_trace_command_error=maximum_recovered,
        measured_swing_source_public_rows=4800, CPU_measured_swing_source_baseline_error=measured_error,
        original_action_tolerance=5e-4, original_arm_target_limit_rad_s=5., float_rate_check_tolerance_rad_s=.0005,
        source_policy_sha256=sha(POLICY), source_trace_sha256=sha(TRACE), seed_table_sha256=seed['table_sha256'],
        factory_sha256=sha(path), checker_sha256=sha(Path(__file__)),
        public_phase773_latched_through_grounded_phase774=True,
        future_measured_early_lateral_source_validation_pending=True,
        candidate_native_performance_not_inherited_from_seed=True,
        native_reexecution_of_proposed_policy=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    text = json.dumps(result, indent=2) + '\n'
    with (HERE / 'CPU_PRIOR_CHECK.json').open('x') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
