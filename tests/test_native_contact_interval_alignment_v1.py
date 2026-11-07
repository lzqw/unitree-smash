"""Prevent a preceding command's endpoint from contaminating impulse evidence."""
import unittest
import numpy as np
from scripts.audit_native_contact_microtrace_v1 import interval_indices


class ContactIntervalAlignment(unittest.TestCase):
    def test_previous_command_force_is_excluded(self):
        frames = np.arange(73600, 73761)
        force = np.full(161, 2.)
        force[0] = 10000.  # This belongs to the preceding command interval.
        start, stop, selected = interval_indices(frames, 460)
        self.assertEqual((start, stop), (0, 160))
        self.assertAlmostEqual(force[selected].sum()*.0000625, .02)

    def test_consecutive_intervals_do_not_double_count_boundary(self):
        frames = np.arange(73600, 73921)
        a = interval_indices(frames, 460)[2]
        b = interval_indices(frames, 461)[2]
        self.assertEqual(len(set(frames[a]) & set(frames[b])), 0)
        self.assertEqual(len(set(frames[a]) | set(frames[b])), 320)

    def test_missing_physics_sample_is_rejected(self):
        frames = np.delete(np.arange(73600, 73921), 80)
        with self.assertRaises(ValueError):
            interval_indices(frames, 460)


if __name__ == '__main__':
    unittest.main()
