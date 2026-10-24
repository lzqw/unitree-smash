import unittest
import torch

from quadsmash.control.full_racket_arm_search_v4 import arm_velocity, extend_arm_velocity_parameters
from quadsmash.control.full_racket_recovery_search_v4 import (
    extend_recovery_parameters, recovery_parameters, RECOVERY_LOW, RECOVERY_HIGH)


class RecoverySearchTest(unittest.TestCase):
    def setUp(self):
        self.reference = torch.zeros(32)
        self.reference[7] = .8
        self.reference[26:32] = torch.arange(6) * .5
        self.recovery = torch.tensor([.27, .005, -.27, -.40, .39])

    def test_extension_preserves_every_existing_coordinate_without_aliasing(self):
        extended = extend_recovery_parameters(self.reference, self.recovery)
        self.assertEqual(extended.shape, (37,))
        self.assertTrue(torch.equal(extended[:32], self.reference))
        self.assertTrue(torch.equal(extended[32:], self.recovery))
        extended[0] = 1
        self.assertEqual(self.reference[0], 0)
        again = extend_recovery_parameters(extended, self.recovery)
        self.assertTrue(torch.equal(again, extended))
        again[0] = 2
        self.assertEqual(extended[0], 1)

    def test_new_standing_coordinates_do_not_replace_arm_velocities(self):
        extended = extend_recovery_parameters(self.reference, self.recovery)
        self.assertTrue(torch.equal(arm_velocity(extended[None], torch.zeros(6)),
                                    arm_velocity(self.reference[None], torch.zeros(6))))
        self.assertTrue(torch.equal(extend_arm_velocity_parameters(extended, torch.zeros(6)), extended))

    def test_standing_uses_lane_specific_searched_values_or_exact_legacy_fallback(self):
        old = self.reference[None].repeat(2, 1)
        self.assertTrue(torch.equal(recovery_parameters(old, self.recovery), self.recovery[None].repeat(2, 1)))
        new = extend_recovery_parameters(self.reference, self.recovery)[None].repeat(2, 1)
        new[1, 36] = .9
        self.assertTrue(torch.equal(recovery_parameters(new, self.recovery), new[:, 32:37]))
        self.assertEqual(recovery_parameters(new, self.recovery)[1, -1], .9)

    def test_unknown_layout_nonfinite_and_out_of_bound_policy_values_are_rejected(self):
        for width in [26, 31, 33, 36, 38]:
            with self.assertRaises(ValueError):
                extend_recovery_parameters(torch.zeros(width), self.recovery)
        for recovery in [torch.tensor([float('nan')]*5), torch.tensor(RECOVERY_LOW)-.01,
                         torch.tensor(RECOVERY_HIGH)+.01]:
            with self.assertRaises(ValueError):
                extend_recovery_parameters(self.reference, recovery)
        with self.assertRaises(ValueError):
            recovery_parameters(torch.zeros(2, 36), self.recovery)


if __name__ == '__main__':
    unittest.main()
