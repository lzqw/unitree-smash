"""CPU control-flow contracts only; these fixtures do not validate physics."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('REPAIR_CONTRACT_SOURCE', str(
    ROOT / 'scripts/pure_control_state_driven_long_mpc_v1_qp_continuity_repair.py')))
spec = importlib.util.spec_from_file_location('repair_contract_subject', SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class FixtureController(subject.PureControlStateDrivenLongMPCV1QPContinuityRepair):
    def __init__(self):
        self.interface = SimpleNamespace(limits=np.tile([-2., 2.], (18, 1)),
                                         previous_command=np.full(18, -.2))
        self.model = SimpleNamespace()
        self.geometry = SimpleNamespace(spheres=[], evaluate=lambda q: (np.array([.1]), None))
        self._tick = 7
        self._plan_cursor = 4
        self._plan = [np.full(18, .4)]
        self.phase = 'BUFFER_LOAD'
        self.reject_future = False
        self.raise_future = False

    def _base_reference(self, state):
        return np.full(18, .2 + .001 * state.step)

    def _make_repair_worker(self):
        return copy.deepcopy(self)

    def solve(self, state, reference):
        # Mimic actual QP binding of interface history; also detect accidental
        # mutations of scheduling state if a worker is not isolated.
        self.interface.previous_command = state.previous_command_target.copy()
        self._tick += 1
        self._plan_cursor += 1
        self.phase = 'MUTATED_IN_TRIAL'
        if self.raise_future:
            raise ValueError('injected numerical rejection')
        return dict(accepted=not self.reject_future, joint_position_target=reference)

    def _risk(self, *args):
        return 0.


class Predictor:
    def __init__(self, fail_call=None):
        self.fail_call = fail_call
        self.calls = []

    def __call__(self, model, interface, state, target, *args, **kwargs):
        self.calls.append(np.array(target))
        next_state = copy.deepcopy(state)
        next_state.step += 1
        return dict(state=next_state, diagnostics={
            'nonfoot_ground_violation_m': .01 if len(self.calls) == self.fail_call else 0.})


class RepairContractTests(unittest.TestCase):
    def setUp(self):
        self.controller = FixtureController()
        self.state = SimpleNamespace(step=0, q=np.zeros(1),
                                     previous_command_target=np.full(18, -.2))
        self.qp = np.full(18, .8)
        self.terminal = np.full(18, -.4)

    def run_rollout(self, predictor):
        with patch.object(subject, 'predict_self_contact_spheres', predictor):
            return self.controller._rollout(self.state, self.qp, self.terminal)

    def assert_live_unchanged(self):
        np.testing.assert_array_equal(self.controller.interface.previous_command, np.full(18, -.2))
        np.testing.assert_array_equal(self.state.previous_command_target, np.full(18, -.2))
        self.assertEqual(self.state.step, 0)
        self.assertEqual((self.controller._tick, self.controller._plan_cursor,
                          self.controller.phase), (7, 4, 'BUFFER_LOAD'))

    def test_normal_sequence_is_original_interpolation_without_repair(self):
        predictor = Predictor()
        sequence, diagnostic = self.run_rollout(predictor)
        expected = [(1 - (i+1)/32) * self.qp + (i+1)/32 * self.terminal for i in range(32)]
        np.testing.assert_array_equal(sequence, expected)
        self.assertFalse(diagnostic['sequence_repaired'])
        self.assertEqual(len(predictor.calls), 32)
        self.assert_live_unchanged()

    def test_first_node_failure_never_repairs_or_returns_prefix(self):
        predictor = Predictor(fail_call=1)
        sequence, diagnostic = self.run_rollout(predictor)
        self.assertIsNone(sequence)
        self.assertFalse(diagnostic['repair_attempted'])
        self.assertEqual(len(predictor.calls), 1)
        self.assert_live_unchanged()

    def test_repair_preserves_actual_interpolated_first_command(self):
        predictor = Predictor(fail_call=3)
        sequence, diagnostic = self.run_rollout(predictor)
        expected_first = 31/32 * self.qp + 1/32 * self.terminal
        np.testing.assert_array_equal(sequence[0], expected_first)
        np.testing.assert_array_equal(predictor.calls[3], expected_first)
        self.assertEqual(len(sequence), 32)
        self.assertEqual(len(predictor.calls), 3 + 32)
        self.assertTrue(diagnostic['sequence_repaired'])
        self.assertTrue(diagnostic['repair_attempted'])

    def test_successful_trial_does_not_mutate_live_history_or_clock(self):
        self.run_rollout(Predictor(fail_call=3))
        self.assert_live_unchanged()

    def test_failed_future_qp_returns_no_partial_plan(self):
        self.controller.reject_future = True
        sequence, diagnostic = self.run_rollout(Predictor(fail_call=3))
        self.assertIsNone(sequence)
        self.assertTrue(diagnostic['repair_attempted'])
        self.assertEqual(diagnostic['repair']['repair_rejection'], 'future_qp_rejected')
        self.assert_live_unchanged()

    def test_numerical_trial_exception_falls_back_without_state_changes(self):
        self.controller.raise_future = True
        sequence, diagnostic = self.run_rollout(Predictor(fail_call=3))
        self.assertIsNone(sequence)
        self.assertEqual(diagnostic['repair']['repair_rejection'], 'repair_exception')
        self.assert_live_unchanged()


if __name__ == '__main__':
    unittest.main()
