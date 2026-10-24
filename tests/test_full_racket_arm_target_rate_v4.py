import unittest
import torch
from quadsmash.control.full_racket_arm_target_rate_v4 import ArmTargetRateV4


class ArmTargetRateTest(unittest.TestCase):
    def test_sudden_targets_and_recovery_reversal_remain_rate_bounded(self):
        rate = ArmTargetRateV4(2, 'cpu', 4.5)
        observed = torch.zeros(2, 6)
        previous = observed.clone()
        for i in range(30):
            desired = torch.full((2, 6), 2. if i < 15 else -2.)
            actual = observed+123.  # Subsequent observations never overwrite the command history.
            if i == 0: actual=observed
            output = rate(desired, actual, torch.full((2,), i*.01))
            self.assertLessEqual(float(abs(output-previous).max()), .045001)
            previous=output

    def test_partial_episode_restart_uses_that_lanes_public_joints_only(self):
        rate = ArmTargetRateV4(2, 'cpu', 4.5)
        desired = torch.ones(2, 6)
        first = rate(desired, torch.zeros(2, 6), torch.tensor([1., 1.]))
        observed = torch.stack([torch.full((6,), -.5), torch.full((6,), 100.)])
        output = rate(desired, observed, torch.tensor([0., 1.01]))
        self.assertTrue(torch.allclose(output[0], torch.full((6,), -.455)))
        self.assertTrue(torch.allclose(output[1], first[1]+.045, atol=1e-6))

    def test_same_clock_does_not_advance_and_long_gap_cannot_jump(self):
        rate = ArmTargetRateV4(1, 'cpu', 5.)
        goal = torch.ones(1, 6)
        initial = rate(goal, torch.zeros_like(goal), torch.tensor([0.]))
        self.assertTrue(torch.equal(initial, rate(goal, goal, torch.tensor([0.]))))
        output = rate(goal, goal, torch.tensor([10.]))
        self.assertLessEqual(float(abs(output-initial).max()), .050001)

    def test_reset_reanchors_to_observed_joints_and_small_targets_are_unchanged(self):
        rate = ArmTargetRateV4(1, 'cpu', 5.)
        desired = torch.full((1, 6), .02)
        self.assertTrue(torch.equal(desired, rate(desired, torch.zeros_like(desired), torch.tensor([0.]))))
        rate.reset()
        observed = torch.full((1, 6), -.1)
        output = rate(desired, observed, torch.tensor([0.]))
        self.assertTrue(torch.allclose(output, torch.full((1, 6), -.05)))

    def test_invalid_rate_and_nonfinite_input_rejected(self):
        for value in [0., -1., 5.1, float('nan')]:
            with self.assertRaises(ValueError):ArmTargetRateV4(1, 'cpu', value)
        rate = ArmTargetRateV4(1, 'cpu', 4.5)
        with self.assertRaises(ValueError):
            rate(torch.full((1,6), float('nan')), torch.zeros(1,6), torch.tensor([0.]))


if __name__ == '__main__':unittest.main()
