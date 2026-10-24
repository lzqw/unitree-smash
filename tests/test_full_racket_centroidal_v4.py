"""Independent momentum checks for the offline floating-base proposal math."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[1] / 'scripts/propose_full_racket_centroidal_velocity_v4.py'
spec = importlib.util.spec_from_file_location('centroidal_proposals', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CentroidalMapsTest(unittest.TestCase):
    def setUp(self):
        self.mass = np.array([4., 1., .3])
        self.positions = np.array([[0., 0., 0.], [.4, .1, .2], [.8, .2, .3]])
        self.inertia = np.array([np.diag([.12, .17, .21]), np.diag([.01, .02, .03]),
                                 np.diag([.002, .003, .004])])
        self.linear = np.zeros((3, 3, 2))
        self.angular = np.zeros((3, 3, 2))
        axes = np.array([[0., 1., 0.], [0., 0., 1.]])
        origins = np.array([[.1, 0., 0.], [.4, .1, .2]])
        for body in [1, 2]:
            for joint in range(body):
                self.linear[body, :, joint] = np.cross(axes[joint], self.positions[body] - origins[joint])
                self.angular[body, :, joint] = axes[joint]

    def maps(self, positions=None, inertia=None, linear=None, angular=None):
        return module.momentum_maps(self.mass, self.positions if positions is None else positions,
                                    self.inertia if inertia is None else inertia,
                                    self.linear if linear is None else linear,
                                    self.angular if angular is None else angular)

    def test_joint_change_conserves_linear_and_angular_momentum(self):
        c, jc, response, locked = self.maps()
        dq = np.array([1.7, -.8])
        omega = response @ dq
        velocities, rates = [], []
        for i in range(3):
            velocities.append(np.cross(omega, self.positions[i] - c) + self.linear[i] @ dq - jc @ dq)
            rates.append(omega + self.angular[i] @ dq)
        linear = sum(m * v for m, v in zip(self.mass, velocities))
        angular = sum(self.inertia[i] @ rates[i] + np.cross(self.positions[i] - c,
                                                          self.mass[i] * velocities[i]) for i in range(3))
        np.testing.assert_allclose(linear, 0., atol=1e-14)
        np.testing.assert_allclose(angular, 0., atol=1e-14)
        self.assertLess(omega[1], 0.)
        self.assertTrue((np.linalg.eigvalsh(locked) > 0).all())

    def test_translation_does_not_change_base_response(self):
        c, jc, response, locked = self.maps()
        shifted, j2, b2, i2 = self.maps(positions=self.positions + [42., -31., 9.])
        np.testing.assert_allclose(shifted, c + [42., -31., 9.])
        for original, transformed in [(jc, j2), (response, b2), (locked, i2)]:
            np.testing.assert_allclose(original, transformed, atol=1e-13)

    def test_rigid_rotation_covariance(self):
        c, jc, response, locked = self.maps()
        rot = Rotation.from_rotvec([.3, -.5, .2]).as_matrix()
        c2, j2, b2, i2 = self.maps(positions=self.positions @ rot.T,
                                  inertia=rot @ self.inertia @ rot.T,
                                  linear=rot @ self.linear, angular=rot @ self.angular)
        for expected, actual in [(rot @ c, c2), (rot @ jc, j2), (rot @ response, b2),
                                  (rot @ locked @ rot.T, i2)]:
            np.testing.assert_allclose(expected, actual, atol=1e-14)

    def test_no_relative_motion_has_no_base_response(self):
        _, jc, response, _ = self.maps(linear=np.zeros_like(self.linear), angular=np.zeros_like(self.angular))
        np.testing.assert_array_equal(jc, np.zeros((3, 2)))
        np.testing.assert_array_equal(response, np.zeros((3, 2)))

    def test_contact_velocity_map_matches_independent_velocity_sum(self):
        c, jc, response, _ = self.maps()
        hit = np.array([1., -.1, .4])
        jp = np.array([[.2, .1], [0., .4], [-.5, 0.]])
        dq = np.array([-.6, 2.1])
        mapped = (jp - jc - module.skew(hit - c) @ response) @ dq
        independent = jp @ dq - jc @ dq + np.cross(response @ dq, hit - c)
        np.testing.assert_allclose(mapped, independent, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
