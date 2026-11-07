import unittest
import numpy as np
from scripts.terminal_settling_constraints_v2 import component_constraints
from scripts.terminal_settling_constraints_v1 import TARGETS


class ComponentSettling(unittest.TestCase):
    def test_same_feasible_set_as_original_absolute_constraints(self):
        rng = np.random.default_rng(618)
        for _ in range(100):
            gap = rng.uniform(-.02, .02, 18)
            velocity = rng.uniform(-.5, .5, 18)
            omega, horizontal, vertical = rng.uniform(0., .4, 3)
            old = np.r_[abs(gap)/TARGETS['command_preload_gap_rad']-1,
                max(abs(velocity))/TARGETS['joint_speed_rad_s']-1,
                omega/TARGETS['angular_speed_rad_s']-1,
                horizontal/TARGETS['horizontal_speed_m_s']-1,
                vertical/TARGETS['vertical_speed_m_s']-1]
            new = component_constraints(gap, velocity, omega, horizontal, vertical)
            self.assertAlmostEqual(new.max(), old.max(), places=13)

    def test_nonmaximal_joint_retains_its_local_constraint_derivative(self):
        velocity = np.zeros(18); velocity[0] = .4; velocity[1] = .3
        epsilon = 1e-6
        before = component_constraints(np.zeros(18), velocity, 0., 0., 0.)
        velocity[1] += epsilon
        after = component_constraints(np.zeros(18), velocity, 0., 0., 0.)
        self.assertEqual(before.max(), after.max())
        self.assertAlmostEqual((after[37]-before[37])/epsilon, 4., places=8)
        self.assertAlmostEqual((after[55]-before[55])/epsilon, -4., places=8)

    def test_zero_components_are_inside_numeric_bounds(self):
        self.assertTrue(np.all(component_constraints(np.zeros(18), np.zeros(18), 0., 0., 0.) < 0))


if __name__ == '__main__':
    unittest.main()
