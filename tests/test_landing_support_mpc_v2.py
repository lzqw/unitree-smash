import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
