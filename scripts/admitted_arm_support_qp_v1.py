"""Prescribe the arm command that the landing executor can actually send.

This corrects target admission before support optimization. It does not free
arm torques, change motor limits or claim native landing performance.
"""
import numpy as np

from scripts.continuous_support_qp_v1 import ContinuousSupportQP


class AdmittedArmSupportQP(ContinuousSupportQP):
    def solve(self, state, joint_reference, *, static_equilibrium=False):
        requested = np.asarray(joint_reference, dtype=np.float64)
        if requested.shape != (18,) or not np.isfinite(requested).all():
            raise ValueError('Require a finite full18-joint command reference')
        if (np.any(requested < self.interface.limits[:, 0] - 1e-6) or
                np.any(requested > self.interface.limits[:, 1] + 1e-6)):
            raise ValueError('Posture target outside original bounds')
        effective = requested.copy()
        if self.preserve_arm_command:
            q = np.asarray(state.joint_position, dtype=np.float64)
            delta = self.interface.velocity[12:] * self.settings.control_dt_s
            effective[12:] = np.clip(
                requested[12:],
                np.maximum(q[12:] - delta, self.interface.limits[12:, 0]),
                np.minimum(q[12:] + delta, self.interface.limits[12:, 1]))
        plan = super().solve(state, effective, static_equilibrium=static_equilibrium)
        plan.update(arm_reference_admission='original-limits-and-one-control-tick',
                    requested_arm_reference=requested[12:].tolist(),
                    admitted_arm_reference=effective[12:].tolist(),
                    arm_reference_admission_max_delta_rad=float(
                        np.max(np.abs(effective[12:] - requested[12:]))))
        return plan
