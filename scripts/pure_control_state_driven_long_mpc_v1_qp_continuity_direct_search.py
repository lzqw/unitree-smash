"""Direct full-sequence command search after QP and repair rejection.

This is one state-driven rescue mechanism.  It searches two deterministic
full-authority constant-target 32-node command sequences (leg extension
factors 2.25 and 3.0), validates every node with the existing executor-aware
predictor and executes only a sequence that passes all original checks.
No candidate is executed partially; failed searches retain the parent
fallback.  The search does not use case IDs or a reference library.
"""
import copy
import numpy as np

from scripts.preview_recovery_execution_v1 import preview_recovery_execution
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import (
    PureControlStateDrivenLongMPCV1QPContinuityRepair,
)


class PureControlStateDrivenLongMPCV1QPContinuityDirectSearch(
        PureControlStateDrivenLongMPCV1QPContinuityRepair):
    direct_leg_scales = (2.25, 3.0)
    # A rejected full-sequence search remains valid only for the current
    # fallback episode. Repeating the identical 32-node search every 10 ms
    # does not change the candidate and can starve the native rollout.
    direct_search_retry_ticks = 32
    controller_name = 'pure_control_state_driven_long_mpc_v1_qp_continuity_direct_search'

    def __init__(self, interface, nominal_posture):
        super().__init__(interface, nominal_posture)
        self._direct_search_last_attempt_tick = -10**9
        self._direct_search_last_phase = None

    def _direct_goals(self, state):
        base = np.asarray(self._base_reference(state), dtype=float)
        goals = []
        for scale in self.direct_leg_scales:
            goal = base.copy()
            goal[:12] = np.clip(
                state.joint_position[:12] + scale * (self.nominal[:12] - state.joint_position[:12]),
                self.interface.limits[:12, 0], self.interface.limits[:12, 1])
            goals.append((float(scale), goal))
        return goals

    def _direct_rollout(self, state, goal):
        predicted = copy.deepcopy(state)
        sequence = []
        for node in range(self.horizon_steps):
            requested = np.asarray(goal, dtype=float).copy()
            try:
                result = self._predict_request(predicted, requested)
            except (ValueError, FloatingPointError) as exc:
                return None, dict(accepted=False, failed_node=node,
                                  rejection='direct_prediction_exception', error=str(exc))
            accepted, diagnostics, violations = self._accepted_prediction(result)
            if not accepted:
                return None, dict(accepted=False, failed_node=node,
                                  rejection='direct_prediction_rejected',
                                  diagnostics=diagnostics, violations=violations,
                                  first_ground_violation=result.get('first_ground_violation'))
            predicted = result['state']
            sequence.append(requested)
        score = self._risk(predicted, self.geometry, sequence[0], state.previous_command_target)
        return sequence, dict(accepted=True, score=float(score), horizon_steps=self.horizon_steps,
                              horizon_seconds=self.horizon_steps * .01,
                              final_min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])))

    def command(self, state, phase, time_s):
        # A previously accepted direct sequence is consumed by the parent
        # scheduler.  The parent still applies its original final-command
        # preview and native arbiter at the outer interface.
        target, diagnostic = super().command(state, phase, time_s)
        if diagnostic.get('solver_accepted'):
            self._direct_search_last_phase = phase
            return target, diagnostic

        if phase != self._direct_search_last_phase:
            self._direct_search_last_attempt_tick = -10**9
            self._direct_search_last_phase = phase
        if self._tick - self._direct_search_last_attempt_tick < self.direct_search_retry_ticks:
            diagnostic = dict(diagnostic)
            diagnostic.update(
                direct_search_attempted=False,
                direct_search_deferred=True,
                direct_search_retry_ticks=self.direct_search_retry_ticks,
            )
            return target, diagnostic

        self._direct_search_last_attempt_tick = self._tick

        attempts = []
        selected = None
        for scale, goal in self._direct_goals(state):
            worker = self._make_repair_worker()
            worker.phase = phase
            worker.elapsed = self.elapsed
            worker.interface.bind_previous_command(state.previous_command_target)
            sequence, result = worker._direct_rollout(copy.deepcopy(state), goal)
            attempt = dict(scale=scale, **result)
            attempts.append(attempt)
            if sequence is not None:
                if selected is None or result['score'] < selected[2]['score']:
                    selected = (sequence, scale, result)

        diagnostic = dict(diagnostic)
        diagnostic.update(direct_search_attempted=True,
                          direct_search_candidates=len(attempts),
                          direct_search=attempts,
                          direct_search_executed=False)
        if selected is None:
            return target, diagnostic

        sequence, scale, result = selected
        # The first target is intentionally allowed to differ from the failed
        # QP candidate.  The parent scheduler consumes the remainder on later
        # ticks, while each actual dispatch still passes through the original
        # native execution map.
        self._plan = [np.asarray(x, dtype=float).copy() for x in sequence]
        self._plan_cursor = 1
        self._last_plan_tick = self._tick
        target = self._plan[0].copy()
        diagnostic.update(source='state_driven_direct_command_search', solver_accepted=True,
                          direct_search_selected_scale=float(scale),
                          direct_search_selected_score=float(result['score']),
                          direct_search_sequence_complete=True,
                          direct_search_executed=True)
        selected_final, _ = preview_recovery_execution(
            self.interface.limits, target, state.previous_command_target)
        diagnostic['execution_preview'] = dict(
            requested_target=target.tolist(),
            predicted_final_target=selected_final.tolist(),
            previous_final_target=state.previous_command_target.tolist(),
            preview_only=True, actual_arbiter_unchanged=True)
        return target, diagnostic


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuityDirectSearch']
