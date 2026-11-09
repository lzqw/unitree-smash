"""State-driven landing sequence MPC with a longer receding horizon.

This controller has no trajectory/reference-library lookup.  At a replan event
it solves the current full-body interval problem, rolls the resulting command
toward a state-derived support target for a 32-tick (0.32 s) touchdown/load
horizon, and executes only the next command.  It replans every 16 ticks or
whenever the event phase changes.  The native 8 s episode remains the task
acceptance window; this horizon is the dense contact horizon, not a claim that
the approximate model certifies the entire episode.
"""
import copy
import time
import numpy as np

from scripts.pure_control_state_driven_mpc_v1 import PureControlStateDrivenMPCV1
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


class PureControlStateDrivenLongMPCV1(PureControlStateDrivenMPCV1):
    horizon_steps = 32
    replan_stride = 16

    def __init__(self, interface, nominal_posture):
        super().__init__(interface, nominal_posture)
        self._plan = []
        self._plan_cursor = 0
        self._last_plan_tick = -10**9
        self._last_phase = None
        self._tick = 0

    def _reset_plan(self):
        self._plan = []
        self._plan_cursor = 0

    def _rollout(self, state, first_target, terminal_reference):
        predicted = copy.deepcopy(state)
        sequence = []
        for index in range(self.horizon_steps):
            alpha = float(index + 1) / float(self.horizon_steps)
            target = np.clip((1. - alpha) * first_target + alpha * terminal_reference,
                             self.interface.limits[:, 0], self.interface.limits[:, 1])
            try:
                result = predict_self_contact_spheres(
                    self.model, self.interface, predicted, target,
                    self.geometry.spheres, substeps=1, geometry=self.geometry)
            except (ValueError, FloatingPointError) as exc:
                return None, dict(accepted=False, rejection='long_horizon_prediction_failed',
                                  error=str(exc), failed_step=index)
            diagnostics = result['diagnostics']
            if (diagnostics.get('nonfoot_ground_violation_m', 0.) > 1e-6 or
                    diagnostics.get('joint_position_violation_rad', 0.) > 3e-4 or
                    diagnostics.get('joint_velocity_violation_rad_s', 0.) > 3e-4 or
                    diagnostics.get('motor_torque_violation_nm', 0.) > 3e-4):
                return None, dict(accepted=False, rejection='long_horizon_physical_violation',
                                  failed_step=index)
            predicted = result['state']
            predicted.previous_command_target = target.copy()
            sequence.append(target.copy())
        score = self._risk(predicted, self.geometry, sequence[0], state.previous_command_target)
        return sequence, dict(accepted=True, score=score,
                              horizon_steps=self.horizon_steps,
                              horizon_seconds=self.horizon_steps * .01,
                              final_min_gap_m=float(np.min(self.geometry.evaluate(predicted.q)[0])))

    def command(self, state, phase, time_s):
        if phase not in ['AIR_PREPARE', 'BUFFER_LOAD', 'SUSTAIN']:
            raise ValueError('Recovery ownership required')
        began = time.perf_counter()
        if self.started is None:
            self.started = float(time_s)
            self.initial_arm = state.joint_position[12:].copy()
            self.initial_preload = state.previous_command_target - state.joint_position
        self.phase = phase
        self.elapsed = float(time_s) - self.started
        if self.elapsed < 0.:
            raise ValueError('Reset controller on episode changes')
        self.interface.bind_previous_command(state.previous_command_target)
        phase_changed = phase != self._last_phase
        if phase_changed:
            self._reset_plan()
        need_replan = (not self._plan or self._plan_cursor >= len(self._plan) or
                       self._tick - self._last_plan_tick >= self.replan_stride or phase_changed)
        plan_diag = dict(accepted=False, rejection='cached_plan')
        if need_replan:
            base_reference = self._base_reference(state)
            base_reference = np.clip(base_reference, self.interface.limits[:, 0],
                                     self.interface.limits[:, 1])
            plan = self.solve(state, base_reference)
            if plan.get('accepted'):
                first = np.asarray(plan['joint_position_target'], dtype=float)
                terminal = np.asarray(self.nominal, dtype=float).copy()
                terminal[12:] = base_reference[12:]
                sequence, plan_diag = self._rollout(state, first, terminal)
                if sequence is not None:
                    self._plan = sequence
                    self._plan_cursor = 0
                    self._last_plan_tick = self._tick
                else:
                    self._reset_plan()
            else:
                self._reset_plan()
                plan_diag = dict(accepted=False, rejection=plan.get('rejection', 'qp_rejected'))
        if self._plan:
            target = self._plan[min(self._plan_cursor, len(self._plan) - 1)].copy()
            self._plan_cursor += 1
            source = 'state_driven_long_mpc'
            accepted = True
        else:
            low, high = self.interface.admitted_target_bounds(
                state.joint_position, .01)
            target = np.clip(state.previous_command_target -
                             .01 * np.clip(state.joint_velocity, -5., 5.), low, high)
            source = 'state_driven_long_command_hold'
            accepted = False
        self._last_phase = phase
        self._tick += 1
        return target, dict(
            controller='pure_control_state_driven_long_mpc_v1',
            solver_accepted=accepted,
            source=source,
            phase=phase,
            replanned=need_replan,
            plan=plan_diag,
            horizon_steps=self.horizon_steps,
            horizon_seconds=self.horizon_steps * .01,
            reference_library_required=False,
            train_case_required=False,
            full18_state_driven=True,
            fallback_certified=False,
            native_recovery_certified=False,
            valid_expert_labels=False,
            wall_seconds=time.perf_counter() - began,
        )


__all__ = ['PureControlStateDrivenLongMPCV1']
