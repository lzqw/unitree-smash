"""Command/history safeguards only; these tests do not execute native physics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.native_terminal_shadow_v1 import NativeTerminalShadowV1
from scripts.native_terminal_support_v1 import NativeTerminalSupportV1


class ShadowContractTests(unittest.TestCase):
    def make(self):
        obj = NativeTerminalShadowV1.__new__(NativeTerminalShadowV1)
        obj.case = 3
        obj.records = []
        obj.limits = np.tile([-2., 2.], (18, 1))
        obj.shadow_steps = obj.student_dispatch_steps = obj.teacher_steps = 0
        return obj

    def test_proposals_are_never_executed_and_history_is_actual(self):
        obj = self.make()
        commands = torch.linspace(-.5, .5, 128*18).reshape(128, 18)
        original = commands.clone()
        obj.previous = np.full(18, .12)
        prior = obj.previous.copy()

        def fake_proposal(self, tick, public, student, finished, native_state):
            output = student.clone()
            output[self.case] = -.3
            requested = self.decode(output[self.case].numpy())
            self.records.append(dict(final_action=output[self.case].numpy().copy(),
                final_target_rad=requested.copy(), requested_target_rad=requested.copy(),
                previous_target_rad=self.previous.copy(), mode='proposal_only'))
            self.previous = requested.copy()
            self.teacher_steps += 1
            return output

        with patch.object(NativeTerminalSupportV1, 'action', fake_proposal):
            result = obj.action(460, None, commands, None)
            self.assertTrue(torch.equal(result, original))
            self.assertTrue(torch.equal(commands, original))
            self.assertEqual(obj.teacher_steps, 0)
            self.assertEqual(obj.shadow_steps, 1)
            np.testing.assert_array_equal(obj.records[0]['previous_target_rad'], prior)
            np.testing.assert_array_equal(obj.previous, obj.decode(commands[3].numpy()))
            np.testing.assert_array_equal(obj.records[0]['final_action'], commands[3].numpy())
            self.assertFalse(np.array_equal(obj.records[0]['shadow_proposal_action'], commands[3].numpy()))
            actual_previous = obj.previous.copy()
            obj.action(461, None, commands, None)
            np.testing.assert_array_equal(obj.records[1]['previous_target_rad'], actual_previous)
            # The returned tensor has independent storage as in the original adapter.
            result.zero_()
            self.assertTrue(torch.equal(commands, original))

    def test_input_mutation_is_rejected(self):
        obj = self.make()
        def corrupt(self, tick, public, student, finished, native_state):
            student[0, 0] = 1.
            return student
        with patch.object(NativeTerminalSupportV1, 'action', corrupt):
            with self.assertRaisesRegex(ValueError, 'modified'):
                obj.action(460, None, torch.zeros(128, 18), None)

    def test_incomplete_shadow_cannot_create_completed_result(self):
        obj = self.make()
        obj.started = False
        class Planner:
            closed = False
            def close(self): self.closed = True
        obj.planner = Planner()
        with tempfile.TemporaryDirectory() as folder:
            obj.output = Path(folder)
            with self.assertRaisesRegex(ValueError, 'full unchanged'):
                obj.save()
            self.assertFalse((obj.output / 'sequence_native_result.json').exists())
            self.assertTrue(obj.planner.closed)


if __name__ == '__main__':
    unittest.main()
