import unittest
import torch
from quadsmash.control.full_racket_face_feedback_v4 import FullRacketFaceFeedback, validate_face_feedback
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


class FaceFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.model = FullRacketKinematicsV4().double()
        self.feedback = FullRacketFaceFeedback({}).double()
        self.limits = torch.tensor(FullRacketGeometryV4().bounds)
        # Actual CEM12generation0env0strings-contact joints, not a new rollout.
        self.q = torch.tensor([[-.5438749194, 1.6235225201, -2.5918354988,
                                .8267433643, .9035215378, 2.0921623707]], dtype=torch.float64)
        self.offset = torch.tensor([.0256, 0., .1], dtype=torch.float64)

    def test_measured_contact_normal_is_corrected_toward_the_public_world_goal(self):
        target = torch.tensor([[.9921963871, .1049016306, .0673941934]], dtype=torch.float64)
        point, normal, *_ = self.model(self.q, self.offset)
        delta = self.feedback(self.q, target, self.limits, self.model, self.offset)
        corrected = self.q + delta
        new_point, new_normal, *_ = self.model(corrected, self.offset)
        before = torch.linalg.vector_norm(target - normal)
        after = torch.linalg.vector_norm(target - new_normal)
        self.assertLess(float(after), float(before) * .25)
        self.assertLess(float(torch.linalg.vector_norm(new_point - point)), .015)
        self.assertLessEqual(float(delta.abs().max()), .4000001)
        self.assertTrue(bool((corrected >= self.limits[:, 0]).all()))
        self.assertTrue(bool((corrected <= self.limits[:, 1]).all()))

    def test_matching_normal_keeps_the_original_joint_target(self):
        normal = self.model(self.q, self.offset)[1]
        delta = self.feedback(self.q, normal, self.limits, self.model, self.offset)
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-12, rtol=0)

    def test_opposite_face_and_singular_pose_stay_finite_and_bounded(self):
        actual = torch.zeros_like(self.q).clamp(self.limits[:, 0], self.limits[:, 1])
        normal = self.model(actual, self.offset)[1]
        delta = self.feedback(actual, -normal, self.limits, self.model, self.offset)
        self.assertTrue(bool(torch.isfinite(delta).all()))
        self.assertLessEqual(float(delta.abs().max()), .4000001)

    def test_invalid_settings_are_rejected(self):
        self.assertIsNone(validate_face_feedback(None))
        for settings in [{'iterations': 0}, {'iterations': 1.5}, {'damping': 0},
                         {'normal_weight': float('nan')}, {'maximum_joint_delta_rad': 3.}]:
            with self.assertRaises(ValueError):
                validate_face_feedback(settings)


if __name__ == '__main__':
    unittest.main()
