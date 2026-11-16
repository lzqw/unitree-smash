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
from scripts.full_authority_position_interface_v1 import FullAuthorityPositionPDInterfaceV1
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity import (
    PureControlStateDrivenLongMPCV1QPContinuity,
)


class PureControlStateDrivenLongMPCV1QPContinuityRepair(
        PureControlStateDrivenLongMPCV1QPContinuity):
    """Preserve the baseline unless a complete future-sequence repair works."""

    controller_name = 'pure_control_state_driven_long_mpc_v1_qp_continuity_repair'

    # One frozen, deterministic direct-search contract.  The search is only
    # entered after both the normal interpolation and the per-node repair have
    # failed.  Five control nodes generate all 32 commands, and every sample
    # perturbs all 18 absolute joint targets.  No TRAIN/case lookup is used.
    direct_search_nodes = (0, 8, 16, 24, 31)
    direct_search_samples = 8
    direct_search_iterations = 1
    direct_search_seed = 20261111
    direct_search_amplitude_rad = 0.08
    direct_search_substeps = 1

    def _bind_actual_history(self, state):
        """Transfer the live final command into an isolated prediction interface.

        The full-authority interface deliberately refuses to infer command
        history from measured joint position.  Search workers and every
        predicted node therefore bind the command that was actually sent at
        the start of that node before querying bounds or previewing execution.
        """
        previous = np.asarray(getattr(state, 'previous_command_target', None),
                              dtype=float)
        if previous.shape != (18,) or not np.isfinite(previous).all():
            raise ValueError('Finite actual previous final command required')
        binder = getattr(self.interface, 'bind_previous_command', None)
        if binder is not None:
            binder(previous)
        return previous.copy()

    def _full_authority_preview(self):
        return isinstance(self.interface, FullAuthorityPositionPDInterfaceV1)

    def _predict_request(self, state, requested):
        previous = self._bind_actual_history(state)
        final, _ = preview_recovery_execution(
            self.interface.limits, requested, previous,
            software_arm_slew=not self._full_authority_preview())
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
        previous = self._bind_actual_history(state)
        final, _ = preview_recovery_execution(
            self.interface.limits, requested, previous,
            software_arm_slew=False)
        diagnostic['execution_preview'] = dict(
            requested_target=requested.tolist(), predicted_final_target=final.tolist(),
            previous_final_target=previous.tolist(),
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

    @staticmethod
    def _sequence_interpolate(control_nodes, node_times, horizon_steps):
        control_nodes = np.asarray(control_nodes, dtype=float)
        node_times = np.asarray(node_times, dtype=int)
        if control_nodes.shape != (len(node_times), 18):
            raise ValueError('Direct-search control nodes must be Kx18')
        sequence = np.empty((horizon_steps, 18), dtype=float)
        for j in range(18):
            sequence[:, j] = np.interp(np.arange(horizon_steps), node_times,
                                       control_nodes[:, j])
        return sequence

    def _direct_search_directions(self):
        """Return deterministic full18 perturbation directions."""
        rng = np.random.default_rng(self.direct_search_seed)
        directions = rng.standard_normal((self.direct_search_samples, len(self.direct_search_nodes), 18))
        norms = np.linalg.norm(directions, axis=2, keepdims=True).clip(min=1e-12)
        return directions / norms

    def _direct_search_score(self, predicted, sequence, previous, min_gap,
                             min_head_gap=float('inf'), min_nonfoot_gap=float('inf')):
        rotation = predicted.root_rotation_world
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        root_velocity = rotation @ predicted.v[:3]
        linear_speed = float(np.linalg.norm(root_velocity))
        angular_speed = float(np.linalg.norm(predicted.v[3:6]))
        command_delta = float(np.mean(np.square(sequence[0] - previous)))
        terminal_support = float(np.count_nonzero(predicted.contacts))
        # All terms are normalized to physical scales before aggregation. The
        # score is used only after a complete 32-node physical pass succeeds.
        # Head and non-foot clearance have separate terms so a large margin on
        # one body group cannot hide a near-ground margin on the other group.
        head_risk = 0. if not np.isfinite(min_head_gap) else max(0., .04 - min_head_gap) / .04
        nonfoot_risk = 0. if not np.isfinite(min_nonfoot_gap) else max(0., .04 - min_nonfoot_gap) / .04
        all_body_risk = max(0., .01 - min_gap) / .01
        return .5 * head_risk ** 2 + .5 * nonfoot_risk ** 2 + \
            .25 * all_body_risk ** 2 + (tilt / .35) ** 2 + \
            (linear_speed / 2.) ** 2 + (angular_speed / 4.) ** 2 + \
            command_delta / (.15 ** 2) - .05 * min(terminal_support, 4.) / 4.

    def _evaluate_direct_sequence(self, state, sequence):
        """Replay one candidate through the original execution and physics checks."""
        predicted = copy.deepcopy(state)
        min_gap = float('inf')
        min_head_gap = float('inf')
        min_nonfoot_gap = float('inf')
        max_com_speed = 0.
        max_angular_speed = 0.
        max_command_step = 0.
        for index, target in enumerate(sequence):
            previous = predicted.previous_command_target.copy()
            try:
                result = self._predict_request(predicted, target)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                return None, dict(accepted=False, failed_step=index,
                                  rejection='direct_search_prediction_failed',
                                  error=repr(exc))
            accepted, diagnostics, violations = self._accepted_prediction(result)
            if not accepted:
                return None, dict(accepted=False, failed_step=index,
                                  rejection='direct_search_physical_violation',
                                  diagnostics=diagnostics, violations=violations)
            predicted = result['state']
            min_gap = min(min_gap, float(np.min(self.geometry.evaluate(predicted.q)[0])))
            gaps = self.geometry.evaluate(predicted.q)[0]
            head = [i for i, shape in enumerate(self.geometry.shapes)
                    if str(shape.get('body', '')).startswith('Head_')]
            nonfoot = [i for i, shape in enumerate(self.geometry.shapes)
                       if not str(shape.get('body', '')).endswith('_foot')]
            if head:
                min_head_gap = min(min_head_gap, float(np.min(gaps[head])))
            if nonfoot:
                min_nonfoot_gap = min(min_nonfoot_gap, float(np.min(gaps[nonfoot])))
            rotation = predicted.root_rotation_world
            max_com_speed = max(max_com_speed, float(np.linalg.norm(rotation @ predicted.v[:3])))
            max_angular_speed = max(max_angular_speed, float(np.linalg.norm(predicted.v[3:6])))
            max_command_step = max(max_command_step,
                                   float(np.max(np.abs(target - previous))))
        score = self._direct_search_score(predicted, sequence,
                                          state.previous_command_target, min_gap,
                                          min_head_gap, min_nonfoot_gap)
        return predicted, dict(accepted=True, full_horizon=True,
            direct_search=True, direct_search_steps=len(sequence), score=float(score),
            final_min_gap_m=float(min_gap), min_head_gap_m=float(min_head_gap),
            min_nonfoot_gap_m=float(min_nonfoot_gap), max_com_speed_m_s=float(max_com_speed),
            max_angular_speed_rad_s=float(max_angular_speed),
            max_command_step_rad=float(max_command_step),
            nodes=list(self.direct_search_nodes))

    def _direct_search_after_both_fail(self, state, nominal_reference):
        """Search full 32-node action sequences from the current real state."""
        previous = self._bind_actual_history(state)
        lower, upper = self.interface.admitted_target_bounds(
            state.joint_position, self.settings.control_dt_s)
        center = np.clip(previous, lower, upper)
        terminal = np.clip(np.asarray(nominal_reference, dtype=float), lower, upper)
        directions = self._direct_search_directions()
        best = None
        evaluated = 0
        feasible = 0
        rejection_counts = {}
        for iteration in range(self.direct_search_iterations):
            for sample in range(self.direct_search_samples):
                # Warm start is sample zero. Other samples move every node,
                # including the first node, so the rescue can change the
                # command that the old repair branch had frozen.
                if sample == 0 and iteration == 0:
                    controls = np.vstack([
                        center,
                        np.linspace(center, terminal, len(self.direct_search_nodes) - 1),
                    ])
                    controls = controls[:len(self.direct_search_nodes)]
                else:
                    envelope = np.linspace(1.0, .35, len(self.direct_search_nodes))[:, None]
                    controls = np.linspace(center, terminal, len(self.direct_search_nodes))
                    controls += self.direct_search_amplitude_rad * envelope * directions[sample]
                    controls = np.clip(controls, lower, upper)
                sequence = self._sequence_interpolate(controls,
                                                      self.direct_search_nodes,
                                                      self.horizon_steps)
                predicted, diagnostic = self._evaluate_direct_sequence(state, sequence)
                evaluated += 1
                if not diagnostic.get('accepted'):
                    key = diagnostic.get('rejection', 'rejected')
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1
                    continue
                feasible += 1
                if best is None or diagnostic['score'] < best[1]['score']:
                    best = (sequence.copy(), diagnostic)
        if best is None:
            return None, dict(accepted=False, direct_search=True,
                              direct_search_attempted=True,
                              direct_search_evaluated=evaluated,
                              direct_search_feasible=feasible,
                              direct_search_rejections=rejection_counts,
                              direct_search_nodes=list(self.direct_search_nodes),
                              direct_search_samples=self.direct_search_samples,
                              direct_search_iterations=self.direct_search_iterations)
        sequence, diagnostic = best
        diagnostic.update(direct_search_attempted=True,
                          direct_search_evaluated=evaluated,
                          direct_search_feasible=feasible,
                          direct_search_rejections=rejection_counts,
                          direct_search_samples=self.direct_search_samples,
                          direct_search_iterations=self.direct_search_iterations,
                          direct_search_seed=self.direct_search_seed,
                          first_request=sequence[0].tolist(),
                          first_final=preview_recovery_execution(
                              self.interface.limits, sequence[0], previous,
                              software_arm_slew=False)[0].tolist())
        return sequence, diagnostic

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

        # A first-node failure has no fixed prefix. The direct search may still
        # change the first command because it starts from the real state.
        if not sequence:
            try:
                worker = self._make_repair_worker()
                searched, search_diag = worker._direct_search_after_both_fail(
                    state, terminal_reference)
            except Exception as exc:
                searched = None
                search_diag = dict(accepted=False, direct_search=True,
                                   direct_search_attempted=True,
                                   direct_search_rejection='direct_search_exception',
                                   error_type=type(exc).__name__, error=str(exc))
            if searched is not None:
                search_diag.update(original_failure=original_failure,
                                   original_prefix_steps=0,
                                   repair_attempted=False,
                                   sequence_direct_searched=True)
                return searched, search_diag
            return None, dict(original_failure, sequence_repaired=False,
                              repair_attempted=False, sequence_direct_searched=False,
                              direct_search=search_diag, prefix_steps=0)

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
        # Both existing branches failed.  This is the only place where the
        # new direct sequence optimizer may run; normal and repaired paths are
        # returned unchanged above.
        try:
            worker = self._make_repair_worker()
            searched, search_diag = worker._direct_search_after_both_fail(
                state, terminal_reference)
        except Exception as exc:
            searched = None
            search_diag = dict(accepted=False, direct_search=True,
                               direct_search_attempted=True,
                               direct_search_rejection='direct_search_exception',
                               error_type=type(exc).__name__, error=str(exc))
        if searched is not None:
            search_diag['original_failure'] = original_failure
            search_diag['original_prefix_steps'] = len(sequence)
            search_diag['repair_attempted'] = True
            search_diag['sequence_repaired'] = False
            search_diag['sequence_direct_searched'] = True
            return searched, search_diag
        return None, dict(
            original_failure,
            sequence_repaired=False,
            repair_attempted=True,
            sequence_direct_searched=False,
            repair=repair_diag,
            prefix_steps=len(sequence),
        )


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuityRepair']
