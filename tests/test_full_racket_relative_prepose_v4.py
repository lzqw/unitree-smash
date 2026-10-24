import unittest

import numpy as np

from quadsmash.control.full_racket_relative_prepose_v4 import (
    public_target_body, relative_prepose_delta, validate_relative_prepose)


class RelativePreposeTest(unittest.TestCase):
    def setUp(self):
        self.public = np.zeros((2, 794))
        self.public[:, 8] = -1.
        self.public[:, 763] = 1.
        self.public[:, 760:763] = [1.10, 0., 1.80]
        self.target = np.tile([1.05, -.14, 1.78], (2, 1))
        self.jp = np.tile(np.c_[np.eye(3), np.zeros((3, 3))], (2, 1, 1))
        self.jn = np.zeros((2, 3, 6))
        self.lower, self.upper = np.full((2, 6), -.3), np.full((2, 6), .3)
        self.reference = public_target_body(self.target[:1], self.public[:1])[0][0]

    def solve(self, gain=1., valid=None):
        return relative_prepose_delta(self.target, self.public, self.reference, self.jp, self.jn,
                                      [True, True] if valid is None else valid, gain, self.lower, self.upper)

    def test_identical_causal_reference_has_exactly_zero_correction(self):
        for gain in [.2, .5, 1.]:
            np.testing.assert_array_equal(self.solve(gain), np.zeros((2, 6)))

    def test_serialized_reference_and_native_float32_axes_agree_exactly(self):
        self.public[:, 6:9] = [-.0321734, .0528534, -.9980832]
        self.public[:, 763:766] = [.9982225, -.0514567, -.0314241]
        native = self.public.astype(np.float32)
        serialized = native.astype(np.float64)
        expected, _ = public_target_body(self.target, serialized)
        actual, _ = public_target_body(self.target, native)
        np.testing.assert_array_equal(actual, expected)

    def test_current_root_translation_and_ball_shift_both_change_preparation(self):
        self.public[0, 760] -= .08
        self.target[1, 1] += .12
        correction = self.solve()
        expected = np.zeros((2, 6))
        expected[0, 0] = -.08/(1+.04**2)
        expected[1, 1] = .12/(1+.04**2)
        np.testing.assert_allclose(correction, expected, atol=1e-14)

    def test_yaw_and_tilt_use_observed_body_axes(self):
        self.public[1, 763:766] = [0., -1., 0.]
        self.public[1, 760:763] = [0., -1.10, 1.80]
        body, valid = public_target_body(self.target, self.public)
        np.testing.assert_allclose(body[1], [-.14, -1.05, 1.78])
        self.assertTrue(valid.all())

    def test_gain_zero_invalid_forecasts_and_invalid_axes_have_no_effect(self):
        self.target[0, 0] = np.nan
        self.public[1, 763:766] = 0.
        np.testing.assert_array_equal(self.solve(), np.zeros((2, 6)))
        self.setUp()
        self.target[:, 0] += 1.
        np.testing.assert_array_equal(self.solve(0.), np.zeros((2, 6)))
        np.testing.assert_array_equal(self.solve(valid=[False, False]), np.zeros((2, 6)))

    def test_unused_observations_cannot_leak_into_correction(self):
        self.target[:, 0] += .1
        expected = self.solve()
        used = np.r_[np.arange(6, 9), np.arange(760, 766)]
        self.public[:, np.setdiff1d(np.arange(794), used)] = np.nan
        np.testing.assert_array_equal(self.solve(), expected)

    def test_bounded_commands_reallocate_a_saturated_joint_and_leave_inputs_untouched(self):
        self.jp[:] = np.c_[np.eye(3), np.eye(3)]
        self.upper[:, 3] = 0.
        self.target[:, 0] += .1
        before = [x.copy() for x in [self.target, self.public, self.jp, self.upper]]
        correction = self.solve()
        self.assertTrue((correction <= self.upper).all() and (correction >= self.lower).all())
        self.assertGreater(correction[0, 0], .09)
        self.assertLessEqual(correction[0, 3], 1e-12)
        for actual, expected in zip([self.target, self.public, self.jp, self.upper], before):
            np.testing.assert_array_equal(actual, expected)

    def test_reference_validation_rejects_wrong_frame_clock_plane_or_nonfinite_values(self):
        config = dict(reference_target_body_m=[.4, -.1, 1.3], frame='robot_root_body',
                      reference_capture_clock_s=2., reference_intercept_plane_x_m=1.05)
        self.assertEqual(validate_relative_prepose(config), config)
        self.assertEqual(validate_relative_prepose(dict(config, reference_capture_clock_s=2.01))['reference_capture_clock_s'], 2.01)
        self.assertIsNone(validate_relative_prepose(None))
        for patch in [dict(frame='world'), dict(reference_capture_clock_s=2.75),
                      dict(reference_intercept_plane_x_m=2.), dict(reference_target_body_m=[0., np.nan, 1.])]:
            with self.assertRaises(ValueError):
                validate_relative_prepose(dict(config, **patch))
        with self.assertRaises(ValueError):
            self.solve(1.01)


if __name__ == '__main__':
    unittest.main()
