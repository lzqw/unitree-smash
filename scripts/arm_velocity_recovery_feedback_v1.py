"""Measured arm-velocity tracking through the existing position command interface.

This modifies requests, not physical PD gains or limits. A bounded candidate
for coupled-model/native validation, not a certified safety filter.
"""
import numpy as np


class ArmVelocityRecoveryFeedbackV1:
    def __init__(self, policy, reference, interface, *, damping_nm_per_rad_s=1., max_offset_rad=.02):
        if not np.isfinite(damping_nm_per_rad_s) or not 0 <= damping_nm_per_rad_s <= 4:
            raise ValueError('Bounded nonnegative request damping required')
        if not np.isfinite(max_offset_rad) or not 0 < max_offset_rad <= .02:
            raise ValueError('Bounded command offset required')
        self.policy = policy
        self.reference = reference
        self.interface = interface
        self.damping = damping_nm_per_rad_s
        self.max_offset = max_offset_rad
        self.maximum_applied_offset = 0.
        self.clipped_requests = 0

    def action(self, offset, state, previous):
        command, mode = self.policy.action(offset, state, previous)
        reference_velocity = self.reference.v_reference[offset, 12:] if offset < 80 else np.zeros(6)
        error = np.asarray(state.joint_velocity)[12:] - reference_velocity
        correction = -self.damping * error / self.interface.kp[12:]
        if not np.isfinite(correction).all():
            raise ValueError('Nonfinite measured arm-velocity feedback')
        self.clipped_requests += int(np.any(abs(correction) > self.max_offset))
        original = command.copy()
        command[12:] += np.clip(correction, -self.max_offset, self.max_offset)
        command = np.clip(command, self.interface.limits[:, 0], self.interface.limits[:, 1])
        command[12:] = np.clip(command[12:], np.asarray(previous)[12:] - .01*self.interface.arm_rate,
                              np.asarray(previous)[12:] + .01*self.interface.arm_rate)
        self.maximum_applied_offset = max(self.maximum_applied_offset, float(np.max(abs(command-original))))
        return command, mode + '_measured_arm_velocity_feedback'
