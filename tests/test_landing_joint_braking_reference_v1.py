import unittest
from types import SimpleNamespace

import numpy as np

from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1


class Base:
    def __init__(self, target):
        self.target = target
        self.model = None
        self.interface = SimpleNamespace(limits=np.tile([-2., 2.], (18, 1)))

    def action(self, offset, state, previous):
        return self.target.copy(), 'base'


class BrakingTests(unittest.TestCase):
    def test_instantaneous_motor_work_never_added_by_brake(self):
        # Independent physical invariant before PD saturation: the additional
        # position-target torque must oppose current joint velocity.
        rng = np.random.default_rng(41)
        for _ in range(200):
            target = rng.uniform(-2., 2., 18)
            q = rng.uniform(-2., 2., 18); v = rng.uniform(-30., 30., 18)
            policy = LandingJointBrakingReferenceV1(Base(target), gain_s=.08, lookahead_s=.03)
            result, _ = policy.action(100, SimpleNamespace(joint_position=q, joint_velocity=v), target)
            self.assertTrue(np.all((result[:12]-target[:12])*v[:12] <= 1e-12))
            self.assertTrue(np.all((result >= -2.) & (result <= 2.)))
            np.testing.assert_array_equal(result[12:], target[12:])

    def test_handoff_and_disabled_are_exact(self):
        target = np.arange(18)*.01
        state = SimpleNamespace(joint_position=np.full(18, 1.99), joint_velocity=np.ones(18)*10)
        for gain, tick in [(.1, 80), (0., 90)]:
            result, _ = LandingJointBrakingReferenceV1(Base(target), gain_s=gain, lookahead_s=.03).action(tick, state, target)
            np.testing.assert_array_equal(result, target)

    def test_unthreatened_joint_and_retreat_are_untouched(self):
        target = np.zeros(18)
        state = SimpleNamespace(joint_position=np.full(18, 1.99), joint_velocity=-np.ones(18))
        policy = LandingJointBrakingReferenceV1(Base(target), gain_s=.08, lookahead_s=.03)
        result, _ = policy.action(100, state, target)
        np.testing.assert_array_equal(result, target)

    def test_invalid_configuration(self):
        for value in [-.1, np.nan, .31]:
            with self.assertRaises(ValueError):
                LandingJointBrakingReferenceV1(Base(np.zeros(18)), gain_s=value, lookahead_s=.03)


if __name__ == '__main__': unittest.main()
