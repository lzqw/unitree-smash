import copy
from types import SimpleNamespace
import unittest

import torch

from quadsmash.control.persistent_recovery_assistance_v1 import PersistentRecoveryAssistanceV1
from quadsmash.control.shared_target_assistance_v6 import RECOVERY_FIELDS


class NestedSource:
    """Models the actual outer reset restoring a cached parameter population."""
    def __init__(self):
        self.saved = torch.arange(4 * 38, dtype=torch.float32).reshape(4, 38) / 100
        self.reference_controller = SimpleNamespace(
            parameters=self.saved.clone(), base=self.saved[0].clone(),
            teacher=SimpleNamespace(leg_recovery_blend=SimpleNamespace(seconds=0.)))
        self.logs = []

    def reset(self, ids=None):
        if ids is None:
            self.reference_controller.parameters.copy_(self.saved)
        else:
            self.reference_controller.parameters[ids] = self.saved[ids]

    def action(self, observation, default, limits):
        return self.reference_controller.parameters.clone()


class PersistentRecoveryTests(unittest.TestCase):
    settings = dict(recovery_local_time_s=1.55, recovery_actor_gain=.8,
                    recovery_thigh_bias_rad=.25, recovery_front_back_thigh_bias_rad=0.,
                    recovery_calf_bias_rad=-.15, recovery_roll_feedback_gain=-.45,
                    recovery_blend_s=.08)

    def expected(self, source):
        expected = source.saved.clone()
        for name, value in self.settings.items():
            index = RECOVERY_FIELDS[name][0]
            if index is not None:
                expected[:, index] = value
        return expected

    def test_full_reset_preserves_overrides_without_replacing_other_parameters(self):
        source = NestedSource()
        original_base = source.reference_controller.base.clone()
        settings = copy.deepcopy(self.settings)
        controller = PersistentRecoveryAssistanceV1(source, settings)
        settings['recovery_local_time_s'] = 2.
        for _ in range(3):
            source.reference_controller.parameters.add_(.1)
            controller.reset()
            self.assertTrue(torch.equal(controller.action(None, None, None), self.expected(source)))
            self.assertTrue(torch.equal(source.reference_controller.base, original_base))
            self.assertEqual(source.reference_controller.teacher.leg_recovery_blend.seconds, .08)

    def test_partial_reset_keeps_other_lanes_current_plans(self):
        source = NestedSource()
        controller = PersistentRecoveryAssistanceV1(source, self.settings)
        source.reference_controller.parameters.add_(.123)
        before = source.reference_controller.parameters.clone()
        controller.reset(torch.tensor([1, 3]))
        actual = source.reference_controller.parameters
        self.assertTrue(torch.equal(actual[[0, 2]], before[[0, 2]]))
        self.assertTrue(torch.equal(actual[[1, 3]], self.expected(source)[[1, 3]]))
        before = actual.clone()
        controller.reset(torch.tensor([], dtype=torch.long))
        self.assertTrue(torch.equal(actual, before))

    def test_empty_settings_keep_source_behavior(self):
        plain, wrapped = NestedSource(), NestedSource()
        controller = PersistentRecoveryAssistanceV1(wrapped, {})
        plain.reset()
        controller.reset()
        self.assertTrue(torch.equal(plain.action(None, None, None), controller.action(None, None, None)))

    def test_invalid_settings_do_not_mutate_source(self):
        for bad in [{'unknown': 1.}, {'recovery_actor_gain': float('nan')},
                    {'recovery_local_time_s': True}, {'recovery_local_time_s': 9.}]:
            source = NestedSource()
            before = source.reference_controller.parameters.clone()
            with self.assertRaises(ValueError):
                PersistentRecoveryAssistanceV1(source, bad)
            self.assertTrue(torch.equal(source.reference_controller.parameters, before))


if __name__ == '__main__':
    unittest.main()
