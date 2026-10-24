import unittest
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names
from quadsmash.control.full_racket_teacher_v4 import PARAMETER_NAMES
from quadsmash.control.full_racket_arm_search_v4 import VELOCITY_PARAMETER_NAMES
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_PARAMETER_NAMES


class ParameterLayoutTest(unittest.TestCase):
    def test_native_and_preflight_layout_covers_recovery_without_reordering_old_coordinates(self):
        base = controller_parameter_names()
        arm = controller_parameter_names(search_arm_velocity=True)
        recovery = controller_parameter_names(search_arm_velocity=True, search_recovery_parameters=True)
        self.assertEqual(base, PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual'])
        self.assertEqual(len(base), 26)
        self.assertEqual(arm, base+VELOCITY_PARAMETER_NAMES)
        self.assertEqual(len(arm), 32)
        self.assertEqual(recovery[:32], arm)
        self.assertEqual(recovery[32:], RECOVERY_PARAMETER_NAMES)
        self.assertEqual(len(recovery), 37)
        self.assertEqual(len(set(recovery)), 37)

    def test_recovery_layout_cannot_be_requested_without_velocity_coordinates(self):
        with self.assertRaises(ValueError):
            controller_parameter_names(search_recovery_parameters=True)


if __name__ == '__main__':
    unittest.main()
