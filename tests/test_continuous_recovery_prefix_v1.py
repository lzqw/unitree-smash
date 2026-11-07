import unittest
from types import SimpleNamespace
import numpy as np

from scripts.check_continuous_recovery_prefix_v1 import CommandContinuousPrefixV1


class ContinuousPrefixTest(unittest.TestCase):
    def base(self):
        class Base:
            interface = SimpleNamespace(limits=np.tile([-2., 2.], (18, 1)), arm_rate=10.)
            def action(self, offset, state, previous):
                target = np.full(18, .7)
                target[12:] = np.clip(target[12:], previous[12:]-.1, previous[12:]+.1)
                return target, 'nominal'
        return Base()

    def test_exact_first_command_and_original_reference_at_end(self):
        previous = np.full(18, -.4)
        base = self.base(); candidate = CommandContinuousPrefixV1(base, None, previous)
        self.assertTrue(np.allclose(candidate.action(0, None, previous)[0], previous, atol=1e-15))
        for tick in range(1, 81):
            action, _ = candidate.action(tick, None, previous)
            self.assertLessEqual(float(np.max(abs(action[12:]-previous[12:]))), .100000000000001)
            self.assertTrue(np.all(abs(action) <= 2.))
            if tick == 80:
                np.testing.assert_array_equal(action, base.action(tick, None, previous)[0])
            previous = action

    def test_invalid_history_and_phase_rejected(self):
        with self.assertRaises(ValueError): CommandContinuousPrefixV1(self.base(), None, np.full(18, 3.))
        candidate = CommandContinuousPrefixV1(self.base(), None, np.zeros(18))
        with self.assertRaises(ValueError): candidate.action(340, None, np.zeros(18))


if __name__ == '__main__': unittest.main()
