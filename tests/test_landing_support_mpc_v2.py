import unittest
from types import SimpleNamespace

import numpy as np

from scripts.landing_support_mpc_v1 import LandingSupportMPCV1
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2, _support_margin
from scripts.whole_body_support_qp_v1 import PositionPDInterface


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
    def test_touchdown_impact_projection_removes_downward_normal_velocity(self):
        class FakeMPC:
            _project_touchdown_impact = LandingSupportMPCV2._project_touchdown_impact

            def __init__(self):
                self.qp = type('QP', (), {})()
                self.qp.model = type('Model', (), {})()
                self.qp.model.v_indices = np.arange(18) + 6
                self.qp.model.snapshot = lambda q, v: {
                    'mass_matrix': np.eye(24),
                    'foot_jacobian': np.zeros((4, 3, 24)),
                }
                self.settings = type('Settings', (), {
                    'touchdown_impact_restitution': 0.,
                    'touchdown_impact_max_impulse_ns': 25.,
                })()

        fake = FakeMPC()
        state = type('State', (), {})()
        state.q = np.zeros(19); state.v = np.zeros(24); state.v[2] = -1.
        state.joint_velocity = np.zeros(18)
        state.qp = None
        # Foot 0 world-Z velocity is the free-base z velocity.
        fake.qp.model.snapshot = lambda q, v: {
            'mass_matrix': np.eye(24),
            'foot_jacobian': np.pad(np.eye(3, 24), ((0, 9), (0, 0))).reshape(4, 3, 24),
        }
        info = fake._project_touchdown_impact(state, [0], pre_velocity=state.v.copy())
        self.assertTrue(info['applied'])
        self.assertGreaterEqual(info['impulse_ns'][0], 0.)
        self.assertAlmostEqual(info['post_normal_velocity_mps'][0], 0., places=6)

    def test_torque_bounds_can_match_native_one_tick_target_admission(self):
        limits = np.c_[-np.ones(18), np.ones(18)]
        interface = PositionPDInterface(
            limits, np.full(18, 10.), np.zeros(18), np.full(18, 100.),
            np.full(18, 1.), .6)
        q = np.zeros(18); q[1] = .8
        v = np.zeros(18)
        low, high = interface.torque_bounds(q, v, control_dt_s=.1)
        np.testing.assert_allclose(low[:2], np.array([-1., -1.]))
        np.testing.assert_allclose(high[:2], np.array([1., 1.]))
        target = q + high / interface.kp
        np.testing.assert_allclose(target[:2], np.array([.1, .9]))

    def test_command_target_admission_matches_one_step_velocity_bound(self):
        class FakeMPC:
            admit_command_target = LandingSupportMPCV1.admit_command_target

            def __init__(self):
                self.qp = SimpleNamespace(interface=SimpleNamespace(
                    velocity=np.r_[np.full(16, 2.), 4., 4.]))
                self.settings = SimpleNamespace(control_dt_s=.1)

        fake = FakeMPC()
        state = SimpleNamespace(joint_position=np.r_[np.zeros(17), 1.])
        requested = np.r_[np.ones(17), 0.]
        admitted = fake.admit_command_target(state, requested)
        expected = np.r_[np.full(16, .2), .4, .6]
        np.testing.assert_allclose(admitted, expected)

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

    def test_replay_horizon_applies_contact_mode_additions_before_integration(self):
        class FakeMPC:
            replay_horizon = LandingSupportMPCV1.replay_horizon

            def _integrate(self, state, acceleration, plan=None):
                return SimpleNamespace(value=state.value + int(state.contacts[1]),
                                       joint_position=state.joint_position,
                                       contacts=state.contacts.copy())

            def _after_prediction(self, state):
                return state, []

        fake = FakeMPC()
        initial = SimpleNamespace(value=0, joint_position=np.array([0.]),
                                  contacts=np.array([True, False, False, False]))
        horizon = [{'acceleration': [0.], 'torque': [2.],
                    'joint_position_target': [3.], 'contact_mode_additions': [1]}]
        final, additions = fake.replay_horizon(initial, horizon)
        self.assertEqual(final.value, 1)
        self.assertEqual(additions, [])


if __name__ == '__main__':
    unittest.main()
