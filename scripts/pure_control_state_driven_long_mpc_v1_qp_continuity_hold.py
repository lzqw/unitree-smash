"""QP-continuity long MPC with a short load-preserving target hold.

This variant keeps the verified 0.32 s / 16-tick timing and the native QP
command-continuity cost. During the predicted touchdown/load window it holds
the first feasible support target for a few nodes before ramping to the
terminal posture. There is no post-solve command blending or case lookup.
"""
import copy
import time
import numpy as np

from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity import (
    PureControlStateDrivenLongMPCV1QPContinuity,
)
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


class PureControlStateDrivenLongMPCV1QPContinuityHold(
        PureControlStateDrivenLongMPCV1QPContinuity):
    """Keep the first support target through initial contact loading."""

    load_hold_steps = 8

    def _rollout(self, state, first_target, terminal_reference):
        predicted = copy.deepcopy(state)
        sequence = []
        hold = min(int(self.load_hold_steps), self.horizon_steps - 1)
        ramp_steps = self.horizon_steps - hold
        for index in range(self.horizon_steps):
            if index < hold:
                target = np.asarray(first_target, dtype=float).copy()
            else:
                alpha = float(index - hold + 1) / float(ramp_steps)
                target = ((1. - alpha) * first_target
                          + alpha * terminal_reference)
                target = np.clip(target, self.interface.limits[:, 0],
                                 self.interface.limits[:, 1])
            try:
                result = predict_self_contact_spheres(
                    self.model, self.interface, predicted, target,
                    self.geometry.spheres, substeps=1, geometry=self.geometry)
            except (ValueError, FloatingPointError) as exc:
                return None, dict(accepted=False,
                                  rejection='load_hold_prediction_failed',
                                  error=str(exc), failed_step=index)
            diagnostics = result['diagnostics']
            if (diagnostics.get('nonfoot_ground_violation_m', 0.) > 1e-6 or
                    diagnostics.get('joint_position_violation_rad', 0.) > 3e-4 or
                    diagnostics.get('joint_velocity_violation_rad_s', 0.) > 3e-4 or
                    diagnostics.get('motor_torque_violation_nm', 0.) > 3e-4):
                return None, dict(accepted=False,
                                  rejection='load_hold_physical_violation',
                                  failed_step=index)
            predicted = result['state']
            predicted.previous_command_target = target.copy()
            sequence.append(target.copy())
        score = self._risk(predicted, self.geometry, sequence[0],
                           state.previous_command_target)
        return sequence, dict(
            accepted=True,
            score=score,
            horizon_steps=self.horizon_steps,
            horizon_seconds=self.horizon_steps * .01,
            load_hold_steps=hold,
            load_hold_seconds=hold * .01,
            final_min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])),
        )

    def command(self, state, phase, time_s):
        began = time.perf_counter()
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic = dict(diagnostic)
        diagnostic.update(
            controller='pure_control_state_driven_long_mpc_v1_qp_continuity_hold',
            load_hold_steps=int(self.load_hold_steps),
            post_solve_command_blend=False,
        )
        diagnostic['wall_seconds'] = time.perf_counter() - began
        return target, diagnostic


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuityHold']
