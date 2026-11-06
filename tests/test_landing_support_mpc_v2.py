import unittest
from types import SimpleNamespace

import numpy as np

from scripts.landing_support_mpc_v1 import LandingSupportMPCV1
from scripts.landing_support_mpc_v2 import _support_margin


class SupportMarginTests(unittest.TestCase):
    def test_inside_rectangle_has_positive_margin(self):
        feet = np.array([[-.3, -.2], [.3, -.2], [.3, .2], [-.3, .2]])
        self.assertAlmostEqual(_support_margin([0., 0.], feet), .2, places=6)

    def test_outside_rectangle_has_negative_margin(self):
        feet = np.array([[-.3, -.2], [.3, -.2], [.3, .2], [-.3, .2]])
        self.assertLess(_support_margin([.4, .0], feet), 0.)

    def test_two_foot_margin_is_distance_to_line(self):
        feet = np.array([[-.3, 0.], [.3, 0.]])
        self.assertAlmostEqual(_support_margin([0., .05], feet), -.05, places=6)

    def test_no_contact_is_infeasible(self):
        self.assertEqual(_support_margin([0., 0.], np.empty((0, 2))), -np.inf)


class ReplaySemanticsTests(unittest.TestCase):
    def test_replay_horizon_uses_executor_plan_and_contact_hook(self):
        class FakeMPC:
            replay_horizon = LandingSupportMPCV1.replay_horizon

            def __init__(self):
                self.calls = []

            def _integrate(self, state, acceleration, plan=None):
                self.calls.append(('integrate', plan is not None,
                                   tuple(np.asarray(plan['torque']).tolist())))
                return SimpleNamespace(value=state.value + 1, joint_position=state.joint_position)

            def _after_prediction(self, state):
                self.calls.append(('contact', state.value))
                return state, [state.value]

        fake = FakeMPC()
        horizon = [{'acceleration': [0.], 'torque': [2.], 'joint_position_target': [3.]}]
        initial = SimpleNamespace(value=0, joint_position=np.array([0.]))
        final, additions = fake.replay_horizon(initial, horizon)
        self.assertEqual(final.value, 1)
        self.assertEqual(additions, [1])
        self.assertEqual(fake.calls, [('integrate', True, (2.0,)), ('contact', 1)])


if __name__ == '__main__':
    unittest.main()
