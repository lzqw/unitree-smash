"""Check bounded past-contact planning memory on4800 actual public inputs."""
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
POLICY = MAIN / 'reports/G07_full_racket_v4/gain_body8_sequential_n128_01/best_candidate.pt'
sys.path.insert(0, str(ROOT))
from scripts.initialize_centered_policy_v4 import same_tree


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('observed_contact_guard_factory', HERE / 'controller.py')
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    source = torch.load(POLICY, map_location='cpu', weights_only=False)
    before = copy.deepcopy(source)
    inputs = json.loads((RUN / 'INPUTS.json').read_text())
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    with np.load(RUN / 'public_and_actions_000.npz') as data:
        public, native, valid = [data[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
    baseline = factory.ContactMemoryController(source, 6, 'cpu', 0.)
    expected = []
    with torch.inference_mode():
        for values in public:
            expected.append(baseline.action({'policy': torch.from_numpy(values)}, default, limits).clone())
    results = []
    for seconds in [0., .4, .6]:
        controller = factory.ContactMemoryController(source, 6, 'cpu', seconds)
        native_error = delta = slew = 0.
        previous = None
        seen = torch.zeros(6, dtype=torch.bool)
        changed_before_contact = changed_legs = extended_rows = changed_rows = 0
        first_changed = None
        with torch.inference_mode():
            for values, actual, keep, reference in zip(public, native, valid, expected):
                obs = torch.from_numpy(values.copy())
                original_public = obs.clone()
                seen |= obs[:, 773] > .5
                result = controller.action({'policy': obs}, default, limits)
                assert torch.equal(obs, original_public)
                assert result.shape == (6, 18) and torch.isfinite(result).all() and (abs(result) <= 1).all()
                difference = abs(result - reference)
                if (~seen).any():
                    changed_before_contact += int((difference[~seen] > 0).sum())
                changed_legs += int((difference[:, :12] > 0).sum())
                changed = difference.amax(1) > 1e-7
                changed_rows += int(changed.sum())
                if changed.any() and first_changed is None:
                    first_changed = float(obs[changed, 775].min())
                if controller.guard is not None:
                    extended_rows += int(controller.guard.last_extended.sum())
                    assert not (controller.guard.last_extended & (~seen)).any()
                delta = max(delta, float(difference.max()))
                native_error = max(native_error, float(abs(result.numpy()[keep] - actual[keep]).max()))
                target = limits[:, 0] + .5 * (result + 1) * (limits[:, 1] - limits[:, 0])
                if previous is not None:
                    slew = max(slew, float(abs(target[:, 12:] - previous[:, 12:]).max()) / .01)
                previous = target.clone()
        assert changed_before_contact == changed_legs == 0 and slew <= 5.0005
        if seconds == 0.:
            assert delta == 0. and native_error <= 5e-4
        else:
            assert extended_rows > 0
            other_seen, other_clock = controller.guard.seen[1:].clone(), controller.guard.first_clock[1:].clone()
            controller.reset(torch.tensor([0]))
            assert torch.equal(controller.guard.seen[1:], other_seen)
            assert torch.equal(controller.guard.first_clock[1:], other_clock)
        assert same_tree(source, before)
        row = dict(memory_s=seconds, actual_public_rows=int(valid.sum()), maximum_source_action_difference=delta,
            maximum_original_GPU_action_difference=native_error, changed_public_rows=changed_rows,
            first_changed_clock_s=first_changed, extended_planning_rows=extended_rows,
            changed_before_any_observed_air_contact_values=changed_before_contact, changed_leg_values=changed_legs,
            maximum_arm_target_rate_rad_s=slew, actual794D_observation_bitwise_untouched=True,
            unchanged_source_actor_and_checkpoint_tree=True, selective_reset_other_lanes_unchanged=True,
            new_native_body_recovery_or_rule_outcome_proven=False)
        results.append(row)
        print(json.dumps(row), flush=True)
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), settings_checked=3, results=results,
        source_policy_sha256=sha(POLICY), data_sha256=sha(RUN / 'public_and_actions_000.npz'),
        factory_sha256=sha(HERE / 'controller.py'), checker_sha256=sha(Path(__file__)),
        original_action_tolerance=5e-4, original_arm_rate_limit_rad_s=5.,
        maximum_extra_planning_memory_s=.6, native_rules_physics_target_and_score_unmodified=True,
        native_reexecution=False, formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    with (HERE / 'CPU_CONTROLLER_CHECK.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
