"""Regression cases for retaining MPC constraints beyond a short horizon."""
import unittest
import numpy as np
from scripts.feedback_tail_rollout_v1 import terminal_violations


class FeedbackTailConstraints(unittest.TestCase):
    def setUp(self):
        self.stable = dict(roll=0., pitch=0., horizontal=.02, vertical_speed=.01,
                          omega=.05, joint_speed=.1, support_feet=3, support_margin_m=.03)

    def test_kinematic_rest_does_not_replace_support_margin(self):
        for margin in [None, -.01, .009]:
            with self.subTest(margin=margin):
                row = dict(self.stable, support_margin_m=margin)
                self.assertIn('support_margin_m', terminal_violations(row, .01))

    def test_old_short_horizon_speed_failures_remain_reported(self):
        row = dict(self.stable, joint_speed=3.8511172817914647, omega=1.3153623557166818)
        self.assertEqual(set(terminal_violations(row, .01)), {'joint_speed', 'omega'})

    def test_late_loss_of_attitude_is_not_accepted(self):
        row = dict(self.stable, roll=np.deg2rad(-15.75))
        self.assertIn('roll', terminal_violations(row, .01))

    def test_vertical_motion_and_no_contact_still_fail(self):
        row = dict(self.stable, vertical_speed=.11, support_feet=0)
        self.assertEqual(set(terminal_violations(row, .01)), {'vertical_speed', 'support_feet'})

    def test_finite_stable_support_is_admissible(self):
        self.assertEqual(terminal_violations(self.stable, .01), {})


if __name__ == '__main__':
    unittest.main()
