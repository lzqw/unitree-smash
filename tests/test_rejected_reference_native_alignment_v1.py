"""Check pre/post-command alignment without claiming native physical validation."""
from types import SimpleNamespace
import unittest

import numpy as np
import pinocchio as pin

from scripts.compare_rejected_reference_native_v1 import compare


class RejectedReferenceAlignment(unittest.TestCase):
    def setUp(self):
        model = pin.Model()
        root = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), 'root')
        model.addJoint(root, pin.JointModelRY(), pin.SE3.Identity(), 'joint')
        self.model = SimpleNamespace(model=model, v_indices=np.array([6]))
        q = np.stack([pin.neutral(model) for _ in range(3)])
        q[:, 0] = [0., .1, .3]
        q[:, 7] = [.1, .2, .5]
        v = np.zeros((3, 7)); v[:, 6] = [0., 2., 4.]
        targets = np.array([[.25], [.6], [.7]])
        self.trace = dict(q=q, v=v, final_target_rad=targets)
        self.rows = [dict(control_tick=280+i, q=q[i+1].copy(), v=v[i+1].copy(),
            target=targets[i].copy()) for i in range(2)]

    def test_state_uses_next_tick_and_command_uses_current_tick(self):
        rows = compare(self.model, self.rows, [280, 281, 282], self.trace)
        for row in rows:
            self.assertEqual(row['measured_endpoint_tick'], row['command_tick']+1)
            for key, value in row.items():
                if 'error_' in key or 'difference_' in key:
                    self.assertAlmostEqual(value, 0.)

    def test_pre_command_state_cannot_look_like_zero_error(self):
        wrong = [dict(self.rows[0], q=self.trace['q'][0], v=self.trace['v'][0])]
        row = compare(self.model, wrong, [280, 281, 282], self.trace)[0]
        self.assertAlmostEqual(row['root_position_error_m'], .1)
        self.assertAlmostEqual(row['maximum_joint_position_error_rad'], .1)
        self.assertAlmostEqual(row['maximum_joint_velocity_error_rad_s'], 2.)

    def test_missing_endpoint_and_duplicate_tick_are_rejected(self):
        for ticks in ([280, 281], [280, 280, 282]):
            with self.assertRaises(ValueError):
                compare(self.model, self.rows, ticks, self.trace)


if __name__ == '__main__':
    unittest.main()
