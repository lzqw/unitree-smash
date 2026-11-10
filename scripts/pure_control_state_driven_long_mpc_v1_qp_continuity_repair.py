"""Single-sequence repair for a rejected long-horizon QP rollout.

The native execution transform is unchanged and previewed at every node.
A repair is attempted only
when the first QP target is valid but the authored 32-node interpolation fails
at a later node.  The first target is kept, and each later target is solved
again from the predicted state with the same support QP.  The complete 32-node
sequence must pass the original physical checks; otherwise the parent class
keeps its original fallback.
"""
import copy

import numpy as np

from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.preview_recovery_execution_v1 import preview_recovery_execution
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity import (
    PureControlStateDrivenLongMPCV1QPContinuity,
)


class PureControlStateDrivenLongMPCV1QPContinuityRepair(
        PureControlStateDrivenLongMPCV1QPContinuity):
    """Preserve the baseline unless a complete future-sequence repair works."""

    controller_name = 'pure_control_state_driven_long_mpc_v1_qp_continuity_repair'

    def _predict_request(self, state, requested):
        final, _ = preview_recovery_execution(
            self.interface.limits, requested, state.previous_command_target)
        result = predict_self_contact_spheres(
            self.model, self.interface, state, final,
            self.geometry.spheres, substeps=1, geometry=self.geometry)
        # The next QP and next preview must see the command actually executed,
        # never the unlimited request. The predictor returns a separate state.
        result['state'].previous_command_target = final.copy()
        result['predicted_final_target'] = final.copy()
        return result

    def command(self, state, phase, time_s):
        requested, diagnostic = super().command(state, phase, time_s)
        final, _ = preview_recovery_execution(
            self.interface.limits, requested, state.previous_command_target)
        diagnostic['execution_preview'] = dict(
            requested_target=requested.tolist(), predicted_final_target=final.tolist(),
            previous_final_target=state.previous_command_target.tolist(),
            preview_only=True, actual_arbiter_unchanged=True)
        # Do not return the prelimited command: actual dispatch keeps the
        # original request -> original arbiter -> original PD path.
        return requested, diagnostic

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

    def _make_repair_worker(self):
        # Rebuild native model/geometry scratch data: FCL transforms cannot be
        # deep-copied safely. Only scalar/array controller context is copied.
        worker = type(self)(copy.deepcopy(self.interface), self.nominal.copy())
        worker.settings = copy.deepcopy(self.settings)
        for name in ('started', 'initial_arm', 'initial_preload', 'phase', 'elapsed'):
            setattr(worker, name, copy.deepcopy(getattr(self, name)))
        return worker

    def _repair_after_first(self, state, first_target, terminal_reference):
        """Re-solve all future nodes from the actual state and fixed first node."""
        predicted = copy.deepcopy(state)
        sequence = [np.asarray(first_target, dtype=float).copy()]
        try:
            first = self._predict_request(predicted, sequence[0])
        except (ValueError, FloatingPointError) as exc:
            return None, dict(repair_rejection='first_node_prediction_failed',
                              error=str(exc), repair_steps=0)
        accepted, diagnostics, violations = self._accepted_prediction(first)
        if not accepted:
            return None, dict(repair_rejection='first_node_physical_violation',
                              diagnostics=diagnostics, violations=violations,
                              repair_steps=0)
        predicted = first['state']
        first_final = first['predicted_final_target'].copy()

        for index in range(1, self.horizon_steps):
            # Recompute the existing soft posture preference at the predicted
            # state. No hard terminal posture or fixed interpolation is imposed.
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
                result = self._predict_request(predicted, target)
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
            repaired_first_request=sequence[0].tolist(),
            repaired_first_final=first_final.tolist(),
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
                result = self._predict_request(predicted, target)
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

        # Solves bind command history and mutate model/geometry scratch data.
        # Keep those side effects off the live controller even on rejection.
        try:
            worker = self._make_repair_worker()
            repaired, repair_diag = worker._repair_after_first(
                state, sequence[0], terminal_reference)
        except Exception as exc:
            # Only the optional repair is caught; interrupts still propagate.
            # No candidate command is returned after any solver/worker error.
            repaired = None
            repair_diag = dict(repair_rejection='repair_exception',
                               error_type=type(exc).__name__, error=str(exc))
        if repaired is not None:
            repair_diag['repair_attempted'] = True
            repair_diag['original_failure'] = original_failure
            repair_diag['original_prefix_steps'] = len(sequence)
            original_final, _ = preview_recovery_execution(
                self.interface.limits, sequence[0], state.previous_command_target)
            repair_diag['original_first_request'] = sequence[0].tolist()
            repair_diag['original_first_final'] = original_final.tolist()
            repair_diag['first_request_max_delta_rad'] = float(np.max(abs(repaired[0] - sequence[0])))
            repair_diag['first_final_max_delta_rad'] = float(np.max(abs(
                original_final - np.asarray(repair_diag['repaired_first_final']))))
            return repaired, repair_diag
        return None, dict(
            original_failure,
            sequence_repaired=False,
            repair_attempted=True,
            repair=repair_diag,
            prefix_steps=len(sequence),
        )


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuityRepair']
