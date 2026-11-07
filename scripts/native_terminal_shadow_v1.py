"""Compute the terminal-controller proposal but dispatch the exact student tensor.

This negative control keeps native-state reads and controller calculations,
while withholding every proposed action. It is not a recovery demonstration.
"""
import json
import numpy as np
import torch

from scripts.native_terminal_support_v1 import NativeTerminalSupportV1


class NativeTerminalShadowV1(NativeTerminalSupportV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadow_steps = 0
        self.student_dispatch_steps = 0

    def action(self, tick, public, student, finished, native_state=None):
        before = student.clone()
        record_count = len(self.records)
        proposal = super().action(tick, public, student, finished, native_state)
        if not torch.equal(student, before):
            raise ValueError('Shadow controller modified the supplied student command')
        actual = student[self.case].detach().cpu().numpy().copy()
        actual_target = self.decode(actual)
        if len(self.records) != record_count:
            if len(self.records) != record_count + 1:
                raise ValueError('Unexpected shadow record count')
            row = self.records[-1]
            row['shadow_proposal_action'] = row['final_action'].copy()
            row['shadow_proposal_target_rad'] = row['final_target_rad'].copy()
            row['shadow_requested_target_rad'] = row['requested_target_rad'].copy()
            row['shadow_mode'] = row['mode']
            row['mode'] = 'exact_student_dispatch_shadow_not_executed'
            row['final_action'] = actual.copy()
            row['requested_target_rad'] = actual_target.copy()
            row['final_target_rad'] = actual_target.copy()
            if not np.array_equal(row['shadow_proposal_action'], proposal[self.case].detach().cpu().numpy()):
                raise ValueError('Lost shadow proposal provenance')
            self.shadow_steps += 1
        # The next proposal must use the ACTUAL previous command, not the
        # discarded shadow command or a synthetic continuation of that command.
        self.previous = actual_target.copy()
        self.teacher_steps = 0
        self.student_dispatch_steps += 1
        return student.clone()

    def save(self):
        try:
            if (not self.started or self.applied != 340 or self.shadow_steps != 340
                    or self.student_dispatch_steps != 800 or self.teacher_steps != 0):
                raise ValueError('Require a full unchanged-student shadow experiment')
            numeric = ['q', 'v', 'contacts', 'foot_force_world_n', 'student_action',
                       'final_action', 'previous_target_rad', 'requested_target_rad',
                       'final_target_rad', 'shadow_proposal_action',
                       'shadow_proposal_target_rad', 'shadow_requested_target_rad']
            np.savez_compressed(self.output / 'sequence_native_trace.npz',
                                **{k: np.stack([r[k] for r in self.records]) for k in numeric})
            result = dict(
                teacher_model='terminal_feedback_shadow_not_executed_v1',
                case=self.case, start_tick=self.start, applied_steps=0,
                teacher_steps=0, unverified_student_fallback_steps=0,
                shadow_proposal_steps=self.shadow_steps,
                student_dispatch_steps=self.student_dispatch_steps,
                query_match=self.query_check,
                records=[{k: v for k, v in r.items() if k not in numeric} for r in self.records],
                bundle_sha256=self.manifest['bundle_sha256'],
                validation_path=self.validation_path, validation_sha256=self.validation_sha256,
                warm_plan_sha256=self.warm_plan_sha256, sphere_audit_sha256=self.sphere_audit_sha256,
                native_execution=True, native_shadow_only=True, online_replanning=False,
                sustained_mpc_feedback_implemented=False,
                sustained_terminal_feedback_implemented=False,
                valid_expert_labels=False, automatic_promotion=False,
                scope=__doc__)
            path = self.output / 'sequence_native_result.json'
            with path.open('x') as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
            return result
        finally:
            self.planner.close()
