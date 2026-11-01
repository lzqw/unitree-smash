import types
import unittest

import torch

from quadsmash.control.shared_target_assistance_v6 import apply_recovery_assistance


class AssistanceTests(unittest.TestCase):
    def test_parameter_mapping_source_immutability_and_reset_persistence(self):
        original = torch.arange(40, dtype=torch.float32)
        saved = original.clone()
        reference = types.SimpleNamespace(base=original, parameters=original.repeat(3, 1),
            teacher=types.SimpleNamespace(leg_recovery_blend=types.SimpleNamespace(seconds=.2)))
        settings = dict(recovery_local_time_s=1.55, recovery_thigh_bias_rad=.25,
            recovery_front_back_thigh_bias_rad=0., recovery_calf_bias_rad=-.15,
            recovery_roll_feedback_gain=-.45, recovery_actor_gain=.8, recovery_blend_s=.08)
        apply_recovery_assistance(reference, settings)
        self.assertTrue(torch.equal(original, saved))
        # Mirrors the controller's per-lane reset from its configured base.
        reference.parameters[1] = -100
        reference.parameters[1] = reference.base
        for index, value in ((3, 1.55), (32, .25), (33, 0.), (34, -.15), (35, -.45), (36, .8)):
            self.assertAlmostEqual(float(reference.parameters[1, index]), value, places=6)
        self.assertEqual(float(reference.parameters[1, 37]), 37.)
        self.assertEqual(reference.teacher.leg_recovery_blend.seconds, .08)


if __name__ == '__main__':
    unittest.main()
