"""Single-sequence repair for a rejected long-horizon QP rollout.

The normal 9cb7983 command path is unchanged.  A repair is attempted only
when the first QP target is valid but the authored 32-node interpolation fails
at a later node.  The first target is kept, and each later target is solved
again from the predicted state with the same support QP.  The complete 32-node
sequence must pass the original physical checks; otherwise the parent class
keeps its original fallback.
"""
import copy

import numpy as np

from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity import (
    PureControlStateDrivenLongMPCV1QPContinuity,
)


class PureControlStateDrivenLongMPCV1QPContinuityRepair(
        PureControlStateDrivenLongMPCV1QPContinuity):
    """Preserve the baseline unless a complete future-sequence repair works."""

    controller_name = 'pure_control_state_driven_long_mpc_v1_qp_continuity_repair'

    @staticmethod
    def _violations(diagnostics):
        return {
            name: float(diagnostics.get(name, 0.))
            for name in (
                'nonfoot_ground_violation_m',
                'joint_position_violation_rad',
                'joint_velocity_violation_rad_s',
                'motor_torque_violation_nm',
            )
        }

    @classmethod
    def _accepted_prediction(cls, result):
        diagnostics = dict(result.get('diagnostics') or {})
        violations = cls._violations(diagnostics)
        accepted = not any(
            value > limit for value, limit in zip(
                violations.values(), (1e-6, 3e-4, 3e-4, 3e-4)))
        return accepted, diagnostics, violations

    def _repair_after_first(self, state, first_target, terminal_reference):
        """Re-solve all future nodes from the actual state and fixed first node."""
        predicted = copy.deepcopy(state)
        sequence = [np.asarray(first_target, dtype=float).copy()]
        try:
            first = predict_self_contact_spheres(
                self.model, self.interface, predicted, sequence[0],
                self.geometry.spheres, substeps=1, geometry=self.geometry)
        except (ValueError, FloatingPointError) as exc:
            return None, dict(repair_rejection='first_node_prediction_failed',
                              error=str(exc), repair_steps=0)
        accepted, diagnostics, violations = self._accepted_prediction(first)
        if not accepted:
            return None, dict(repair_rejection='first_node_physical_violation',
                              diagnostics=diagnostics, violations=violations,
                              repair_steps=0)
        predicted = first['state']
        predicted.previous_command_target = sequence[0].copy()

        for index in range(1, self.horizon_steps):
            # The reference is recomputed from the predicted state.  The
            # terminal reference is retained only as a soft posture preference
            # through the existing QP; it is never forced as a hard endpoint.
            reference = np.asarray(self._base_reference(predicted), dtype=float)
            reference = np.clip(
                reference, self.interface.limits[:, 0],
                self.interface.limits[:, 1])
            plan = self.solve(predicted, reference)
            if not plan.get('accepted'):
                return None, dict(
                    repair_rejection='future_qp_rejected',
                    failed_step=index,
                    solver={k: plan.get(k) for k in (
                        'solver_status', 'solver_iterations',
                        'solver_runtime_s', 'residuals', 'rejection')
                        if k in plan},
                    repair_steps=len(sequence))
            target = np.asarray(plan['joint_position_target'], dtype=float)
            try:
                result = predict_self_contact_spheres(
                    self.model, self.interface, predicted, target,
                    self.geometry.spheres, substeps=1, geometry=self.geometry)
            except (ValueError, FloatingPointError) as exc:
                return None, dict(repair_rejection='future_prediction_failed',
                                  failed_step=index, error=str(exc),
                                  repair_steps=len(sequence))
            accepted, diagnostics, violations = self._accepted_prediction(result)
            if not accepted:
                return None, dict(
                    repair_rejection='future_physical_violation',
                    failed_step=index, diagnostics=diagnostics,
                    violations=violations, repair_steps=len(sequence))
            sequence.append(target.copy())
            predicted = result['state']
            predicted.previous_command_target = target.copy()

        score = self._risk(
            predicted, self.geometry, sequence[0], state.previous_command_target)
        return sequence, dict(
            accepted=True,
            full_horizon=True,
            sequence_repaired=True,
            repair_steps=len(sequence),
            horizon_steps=self.horizon_steps,
            horizon_seconds=self.horizon_steps * .01,
            score=score,
            final_min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])),
        )

    def _rollout(self, state, first_target, terminal_reference):
        """Run the original rollout first, then one complete repair attempt."""
        predicted = copy.deepcopy(state)
        sequence = []
        original_failure = None
        for index in range(self.horizon_steps):
            alpha = float(index + 1) / float(self.horizon_steps)
            target = np.clip(
                (1. - alpha) * first_target + alpha * terminal_reference,
                self.interface.limits[:, 0], self.interface.limits[:, 1])
            try:
                result = predict_self_contact_spheres(
                    self.model, self.interface, predicted, target,
                    self.geometry.spheres, substeps=1, geometry=self.geometry)
            except (ValueError, FloatingPointError) as exc:
                original_failure = dict(
                    rejection='long_horizon_prediction_failed',
                    failed_step=index, error=str(exc))
                break
            accepted, diagnostics, violations = self._accepted_prediction(result)
            if not accepted:
                original_failure = dict(
                    rejection='long_horizon_physical_violation',
                    failed_step=index, diagnostics=diagnostics,
                    violations=violations)
                break
            predicted = result['state']
            predicted.previous_command_target = target.copy()
            sequence.append(target.copy())

        if original_failure is None:
            score = self._risk(
                predicted, self.geometry, sequence[0],
                state.previous_command_target)
            return sequence, dict(
                accepted=True, full_horizon=True, sequence_repaired=False,
                score=score, horizon_steps=self.horizon_steps,
                horizon_seconds=self.horizon_steps * .01,
                final_min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])))

        # A failure on the first node has no valid fixed first action, so the
        # parent command path must use its original fallback unchanged.
        if not sequence:
            return None, dict(original_failure, sequence_repaired=False,
                              repair_attempted=False, prefix_steps=0)

        repaired, repair_diag = self._repair_after_first(
            state, first_target, terminal_reference)
        if repaired is not None:
            repair_diag['original_failure'] = original_failure
            repair_diag['original_prefix_steps'] = len(sequence)
            return repaired, repair_diag
        return None, dict(
            original_failure,
            sequence_repaired=False,
            repair_attempted=True,
            repair=repair_diag,
            prefix_steps=len(sequence),
        )


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuityRepair']
