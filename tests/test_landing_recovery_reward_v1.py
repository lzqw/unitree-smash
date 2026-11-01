import unittest

import torch

from quadsmash.control.landing_recovery_reward_v1 import LandingRecoveryRewardV1, DEFAULTS


class LandingRewardTests(unittest.TestCase):
    def inputs(self, count=7):
        yes = lambda: torch.ones(count, dtype=torch.bool)
        no = lambda: torch.zeros(count, dtype=torch.bool)
        zero = lambda: torch.zeros(count)
        return dict(legal_air=yes(), associated_landing=yes(), supported=yes(),
                    robot_fault=no(), body_ground_seen=no(), tilt_rad=zero(),
                    horizontal_speed_mps=zero(), angular_speed_rad_s=zero(),
                    joint_speed_rad_s=zero(), longest_stable_s=zero(), dt=.01)

    def test_only_associated_legal_supported_recovery_is_paid(self):
        reward = LandingRecoveryRewardV1(7, 'cpu', DEFAULTS, .8)
        data = self.inputs()
        for i, name in enumerate(('legal_air', 'associated_landing', 'supported')):
            data[name][i] = False
        data['robot_fault'][3] = True
        data['body_ground_seen'][4] = True
        data['tilt_rad'][5] = float('nan')
        data['longest_stable_s'][:] = .4
        actual = reward.step(**data)
        self.assertTrue(torch.equal(actual[:6], torch.zeros(6)))
        self.assertAlmostEqual(float(actual[6]), 10.06, places=5)

    def test_short_stable_runs_do_not_stack_and_lane_reset_isolated(self):
        reward = LandingRecoveryRewardV1(2, 'cpu', DEFAULTS, .8)
        data = self.inputs(2)
        for stable in (.1, .1, .05, .1, .4, .2, .8, 1.):
            data['longest_stable_s'][:] = stable
            reward.step(**data)
        self.assertAlmostEqual(float(reward.paid_progress), 40., places=5)
        reward.reset(torch.tensor([0]))
        self.assertEqual(float(reward.credited[0]), 0.)
        self.assertEqual(float(reward.credited[1]), 1.)
        data['longest_stable_s'][:] = .4
        reward.step(**data)
        self.assertAlmostEqual(float(reward.paid_progress), 50., places=5)

    def test_quality_rewards_less_motion_and_tilt_without_reference_joints(self):
        reward = LandingRecoveryRewardV1(2, 'cpu', DEFAULTS, .8)
        data = self.inputs(2)
        for name in ('tilt_rad', 'horizontal_speed_mps', 'angular_speed_rad_s', 'joint_speed_rad_s'):
            data[name][1] = 5.
        actual = reward.step(**data)
        self.assertGreater(float(actual[0]), float(actual[1]))
        self.assertAlmostEqual(float(actual[0]), .06, places=6)

    def test_quality_integrates_with_timestep_and_original_window_preserved(self):
        reward = LandingRecoveryRewardV1(1, 'cpu', DEFAULTS, .8)
        data = self.inputs(1)
        for _ in range(100):
            reward.step(**data)
        self.assertAlmostEqual(float(reward.paid_quality), 6., places=5)
        self.assertEqual(float(reward.paid_progress), 0.)
        self.assertEqual(reward.window, .8)


if __name__ == '__main__':
    unittest.main()
