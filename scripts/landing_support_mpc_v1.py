#!/usr/bin/env python3
"""Risk-triggered short-horizon support MPC over the loaded full-body QP.

This is deliberately a small first controller for the current 100 Hz position
interface. It keeps the measured contact mode fixed over one prediction, rolls
the verified full-body support QP forward for a short horizon, and returns the
first position target only when every predicted stage remains feasible. It
does not add forces, teleport state, or claim a native result. A later native
adapter can call ``should_intervene`` from current observations and apply the
returned normalized action through the original mixer.
"""
from dataclasses import dataclass, asdict
import copy
import math

import numpy as np
import pinocchio as pin

from scripts.reduced_whole_body_support_qp_v1 import ReducedWholeBodySupportQP
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


@dataclass(frozen=True)
class LandingMPCSettingsV1:
    horizon_steps: int = 10
    control_dt_s: float = .01
    tilt_trigger_rad: float = .55
    angular_speed_trigger_rad_s: float = 2.2
    horizontal_speed_trigger_mps: float = .75
    minimum_support_feet: int = 2
    terminal_tilt_rad: float = .35
    terminal_angular_speed_rad_s: float = 1.4
    terminal_horizontal_speed_mps: float = .45
    max_stage_runtime_s: float = .02
    executor_replay: bool = False
    # When enabled, replay the realized implicit-PD torque through the same
    # full-body dynamics/contact equations used by the support QP.  The
    # legacy joint-acceleration scaling remains the fallback if the replay
    # solve is ill-conditioned.
    executor_replay_model_consistent: bool = True


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scalar')
    return float(value)


class LandingSupportMPCV1:
    """A receding-horizon wrapper around the existing full-body support QP."""

    def __init__(self, interface: PositionPDInterface, *,
                 qp_settings: SupportSettings = SupportSettings(),
                 settings: LandingMPCSettingsV1 = LandingMPCSettingsV1()):
        if (isinstance(settings.horizon_steps, bool) or not isinstance(settings.horizon_steps, int)
                or settings.horizon_steps < 1 or settings.horizon_steps > 40):
            raise ValueError('horizon_steps must be an integer in 1..40')
        if abs(float(settings.control_dt_s) - float(qp_settings.control_dt_s)) > 1e-12:
            raise ValueError('MPC and support QP control dt must match')
        for name in ('tilt_trigger_rad', 'angular_speed_trigger_rad_s',
                     'horizontal_speed_trigger_mps', 'terminal_tilt_rad',
                     'terminal_angular_speed_rad_s', 'terminal_horizontal_speed_mps',
                     'max_stage_runtime_s'):
            _finite_positive(getattr(settings, name), name)
        if type(settings.executor_replay) is not bool:
            raise ValueError('executor_replay must be boolean')
        if type(settings.executor_replay_model_consistent) is not bool:
            raise ValueError('executor_replay_model_consistent must be boolean')
        if (isinstance(settings.minimum_support_feet, bool)
                or not isinstance(settings.minimum_support_feet, int)
                or not 0 <= settings.minimum_support_feet <= 4):
            raise ValueError('minimum_support_feet must be in 0..4')
        if settings.terminal_tilt_rad > settings.tilt_trigger_rad:
            raise ValueError('terminal tilt must be no larger than trigger tilt')
        self.qp = ReducedWholeBodySupportQP(interface, qp_settings)
        self.settings = settings

    def contract(self):
        return dict(schema='landing-support-mpc-v1', formulation='fixed-contact-sequential-QP-rollout',
                    settings=asdict(self.settings), support_qp_settings=asdict(self.qp.settings),
                    action_interface='original-normalized-position-targets-v1',
                    no_external_force=True, no_state_teleport=True, native_execution=False,
                    solver_failure_is_not_hardware_impossibility=True)

    @staticmethod
    def _tilt_rad(state):
        # state.root_rotation_world maps body to world; body z should align with world z.
        return float(np.arccos(np.clip(state.root_rotation_world[2, 2], -1., 1.)))

    def risk(self, state):
        tilt = self._tilt_rad(state)
        horizontal_speed = float(np.linalg.norm(state.v[:2]))
        angular_speed = float(np.linalg.norm(state.v[3:6]))
        feet = int(np.asarray(state.contacts, dtype=bool).sum())
        trigger = bool(tilt >= self.settings.tilt_trigger_rad
                       or horizontal_speed >= self.settings.horizontal_speed_trigger_mps
                       or angular_speed >= self.settings.angular_speed_trigger_rad_s
                       or feet < self.settings.minimum_support_feet)
        return dict(trigger=trigger, tilt_rad=tilt, horizontal_speed_mps=horizontal_speed,
                    angular_speed_rad_s=angular_speed, support_feet=feet,
                    reason=('support_count' if feet < self.settings.minimum_support_feet else
                            'tilt' if tilt >= self.settings.tilt_trigger_rad else
                            'angular_speed' if angular_speed >= self.settings.angular_speed_trigger_rad_s else
                            'horizontal_speed' if horizontal_speed >= self.settings.horizontal_speed_trigger_mps else
                            'none'))

    def risk_score(self, risk):
        """A local, dimensionless risk score for first-step admission.

        This is deliberately weaker than the terminal support gate.  It is
        only used to reject a first command that makes every local risk
        measure worse; it is not a stability certificate.
        """
        return float(.35 * risk['tilt_rad'] / self.settings.tilt_trigger_rad
                     + .35 * risk['horizontal_speed_mps'] / self.settings.horizontal_speed_trigger_mps
                     + .30 * risk['angular_speed_rad_s'] / self.settings.angular_speed_trigger_rad_s
                     + max(0, self.settings.minimum_support_feet - risk['support_feet']))

    def _integrate(self, state, acceleration, plan=None):
        next_state = copy.deepcopy(state)
        dt = self.settings.control_dt_s
        acceleration = np.asarray(acceleration, dtype=np.float64).copy()
        if self.settings.executor_replay and plan is not None:
            # The QP acceleration is a model request, while Isaac receives a
            # rate-limited normalized position target through implicit PD.  In
            # prediction, replay that interface: limit target motion by the
            # admitted velocity bound, then recompute the torque actually
            # produced by the PD/effort limits and scale only the joint part of
            # the requested generalized acceleration.
            qj = np.asarray(state.joint_position, dtype=np.float64)
            vj = np.asarray(state.joint_velocity, dtype=np.float64)
            target = np.asarray(plan['joint_position_target'], dtype=np.float64)
            max_delta = np.asarray(self.qp.interface.velocity, dtype=np.float64) * dt
            target = qj + np.clip(target-qj, -max_delta, max_delta)
            realized = np.clip(self.qp.interface.kp*(target-qj)-self.qp.interface.kd*vj,
                               -self.qp.interface.effort, self.qp.interface.effort)
            requested = np.asarray(plan['torque'], dtype=np.float64)
            scale = np.divide(realized, requested, out=np.zeros_like(realized),
                              where=np.abs(requested) > 1e-6)
            scale = np.clip(scale, -1., 1.)
            if self.settings.executor_replay_model_consistent:
                replayed = self._replay_realized_dynamics(state, realized)
            else:
                replayed = None
            if replayed is not None:
                acceleration = replayed
            else:
                # Conservative fallback retained for malformed or singular
                # contact snapshots.  It is intentionally not described as a
                # dynamics-consistent replay.
                acceleration[self.qp.model.v_indices] *= scale
        qdot = state.v.copy()
        v = state.v + dt * acceleration
        # Pinocchio free-flyer integration keeps quaternion normalization and
        # joint ordering correct; acceleration is in the model tangent frame.
        next_state.q = pin.integrate(self.qp.model.model, state.q, dt * qdot + .5 * dt * dt * acceleration)
        next_state.v = v
        next_state.joint_position = next_state.q[self.qp.model.q_indices].copy()
        next_state.joint_velocity = next_state.v[self.qp.model.v_indices].copy()
        next_state.root_rotation_world = pin.Quaternion(next_state.q[3:7]).matrix()
        next_state.root_position_world = next_state.q[:3].copy()
        # Contact mode is an observed mode; this first version never invents a
        # new foot contact during its prediction horizon.
        return next_state

    def _replay_realized_dynamics(self, state, realized_torque):
        """Recompute a full-body acceleration for the realized PD torque.

        The QP solves

            M a + h - S tau - J.T f = 0,
            J a = -J v / tau_contact - Jdot v

        for the currently observed contacts.  Re-solving this small KKT
        system after the position-PD interface has clipped the requested
        torque keeps floating-base acceleration, joint acceleration and
        contact reaction consistent during prediction.  This is still a
        model replay (it does not model PhysX impact impulses); returning
        ``None`` lets the caller use the old conservative fallback when the
        linear system is non-finite or ill-conditioned.
        """
        try:
            torque = np.asarray(realized_torque, dtype=np.float64)
            if torque.shape != (18,) or not np.isfinite(torque).all():
                return None
            snapshot = self.qp.model.snapshot(state.q, state.v)
            mass = np.asarray(snapshot['mass_matrix'], dtype=np.float64)
            nonlinear = np.asarray(snapshot['nonlinear'], dtype=np.float64)
            if mass.shape != (24, 24) or nonlinear.shape != (24,):
                return None
            rhs_dynamics = self.qp.selection @ torque - nonlinear
            active = np.flatnonzero(np.asarray(state.contacts, dtype=bool))
            if len(active):
                jac = np.asarray(snapshot['foot_jacobian'][active], dtype=np.float64).reshape(3*len(active), 24)
                jdot_v = (np.asarray(snapshot['foot_jacobian_dot'][active], dtype=np.float64)
                          @ state.v).reshape(3*len(active))
                foot_velocity = jac @ state.v
                contact_acceleration = -foot_velocity / self.qp.settings.contact_velocity_decay_s
                kkt = np.block([
                    [mass, -jac.T],
                    [jac, np.zeros((3*len(active), 3*len(active)))]])
                rhs = np.concatenate([rhs_dynamics, contact_acceleration-jdot_v])
                solution, _, _, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
                acceleration = solution[:24]
                residual = kkt @ solution - rhs
            else:
                acceleration = np.linalg.solve(mass, rhs_dynamics)
                residual = mass @ acceleration - rhs_dynamics
            if (not np.isfinite(acceleration).all() or
                    not np.isfinite(residual).all() or
                    float(np.max(np.abs(residual))) > 2e-5):
                return None
            return acceleration
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return None

    def replay_horizon(self, state, horizon):
        """Replay a stored horizon with the exact prediction semantics.

        Terminal checks and diagnostics must use the same executor replay and
        predicted-contact transition as the rollout that produced the plan.
        """
        predicted = copy.deepcopy(state)
        additions = []
        for row in horizon:
            if 'acceleration' not in row:
                break
            plan = dict(
                acceleration=np.asarray(row['acceleration'], dtype=np.float64),
                torque=np.asarray(row.get('torque', np.zeros(18)), dtype=np.float64),
                joint_position_target=np.asarray(
                    row.get('joint_position_target', predicted.joint_position), dtype=np.float64),
            )
            mode_additions = row.get('contact_mode_additions', [])
            if mode_additions:
                predicted.contacts[np.asarray(mode_additions, dtype=np.int64)] = True
            predicted = self._integrate(predicted, plan['acceleration'], plan)
            predicted, added = self._after_prediction(predicted)
            additions.extend(int(index) for index in added)
        return predicted, additions

    def _after_prediction(self, state):
        """Hook for a derived controller to update predicted contact mode."""
        return state, []

    def _contact_mode_candidates(self, state):
        """Hook for derived controllers to propose imminent contact modes."""
        return []

    def solve(self, state, joint_reference, *, force=False):
        started = __import__('time').perf_counter()
        risk = self.risk(state)
        if not force and not risk['trigger']:
            return dict(accepted=False, intervened=False, risk=risk,
                        rejection='risk trigger not active', horizon=[],
                        native_execution=False, mpc_contract=self.contract())
        # This first controller has a fixed observed contact mode and a
        # support-force QP.  With fewer than the declared minimum support feet
        # it cannot represent touchdown/contact creation, so returning an
        # explicit fallback is safer than asking OSQP to invent a contact
        # transition or treating a free-flight solution as a landing plan.
        if int(np.asarray(state.contacts, dtype=bool).sum()) < self.settings.minimum_support_feet:
            return dict(accepted=False, intervened=True, risk=risk, horizon=[],
                        rejection='insufficient support contacts for fixed-contact QP',
                        solver_failure=False, native_execution=False,
                        wall_runtime_s=__import__('time').perf_counter()-started,
                        mpc_contract=self.contract())
        predicted = copy.deepcopy(state)
        horizon = []
        initial_risk = risk
        initial_score = self.risk_score(initial_risk)
        first_step_risk = None
        first_step_score = None
        for stage in range(self.settings.horizon_steps):
            plan_state = predicted
            mode_additions = []
            plan = self.qp.solve(plan_state, joint_reference)
            if not plan.get('accepted'):
                for candidate_state, candidate_additions in self._contact_mode_candidates(predicted):
                    candidate_plan = self.qp.solve(candidate_state, joint_reference)
                    if candidate_plan.get('accepted'):
                        plan_state = candidate_state
                        mode_additions = list(candidate_additions)
                        plan = candidate_plan
                        break
            row = dict(stage=stage, accepted=bool(plan.get('accepted')), solver_status=plan.get('solver_status'),
                       solver_iterations=plan.get('solver_iterations'), wall_runtime_s=plan.get('wall_runtime_s'),
                       contacts=plan_state.contacts.tolist(), risk=self.risk(plan_state),
                       residuals=plan.get('residuals'), qp_rejection=plan.get('rejection'))
            if not plan.get('accepted'):
                status = str(plan.get('solver_status', '')).lower()
                numerical_failure = not (status.startswith('solved') and
                                         plan.get('residuals') is not None)
                return dict(accepted=False, intervened=True, risk=risk, horizon=horizon + [row],
                            rejection=f'QP rejected prediction stage {stage}', native_execution=False,
                            first_step_risk=first_step_risk,
                            first_step_risk_score=first_step_score,
                            initial_risk_score=initial_score,
                            first_step_improves=False,
                            solver_failure=numerical_failure,
                            constraint_failure=not numerical_failure,
                            wall_runtime_s=__import__('time').perf_counter()-started,
                            mpc_contract=self.contract())
            stage_record = dict(row, residuals=plan.get('residuals'),
                                acceleration=np.asarray(plan['acceleration']).tolist(),
                                torque=np.asarray(plan['torque']).tolist(),
                                normalized_action=plan['normalized_action'].tolist(),
                                joint_position_target=plan['joint_position_target'].tolist())
            if mode_additions:
                stage_record['contact_mode_additions'] = mode_additions
            horizon.append(stage_record)
            predicted = self._integrate(plan_state, plan['acceleration'], plan)
            predicted, added_contacts = self._after_prediction(predicted)
            if added_contacts:
                # ``horizon`` stores a copy; annotate the stored stage record,
                # not the pre-append temporary row, so diagnostics are truthful.
                stage_record['predicted_contact_additions'] = list(added_contacts)
            if stage == 0:
                first_step_risk = self.risk(predicted)
                first_step_score = self.risk_score(first_step_risk)
        first_step_improves = bool(first_step_score is not None and
                                   first_step_score + 1e-6 < initial_score)
        terminal = self.risk(predicted)
        terminal_ok = bool(terminal['tilt_rad'] <= self.settings.terminal_tilt_rad
                           and terminal['angular_speed_rad_s'] <= self.settings.terminal_angular_speed_rad_s
                           and terminal['horizontal_speed_mps'] <= self.settings.terminal_horizontal_speed_mps)
        if not terminal_ok:
            return dict(accepted=False, intervened=True, risk=risk, horizon=horizon,
                        terminal_risk=terminal, rejection='terminal support margin not reached',
                        first_step_risk=first_step_risk,
                        first_step_risk_score=first_step_score,
                        initial_risk_score=initial_score,
                        first_step_improves=first_step_improves,
                        solver_failure=False, native_execution=False,
                        wall_runtime_s=__import__('time').perf_counter()-started,
                        mpc_contract=self.contract())
        first = horizon[0]
        return dict(accepted=True, intervened=True, risk=risk, horizon=horizon,
                    terminal_risk=terminal, normalized_action=np.asarray(first['normalized_action']),
                    joint_position_target=np.asarray(first['joint_position_target']),
                    torque=np.asarray(first['torque']),
                    first_step_risk=first_step_risk,
                    first_step_risk_score=first_step_score,
                    initial_risk_score=initial_score,
                    first_step_improves=first_step_improves,
                    solver_failure=False, native_execution=False,
                    wall_runtime_s=__import__('time').perf_counter()-started,
                    mpc_contract=self.contract())


def make_default_landing_mpc(limits, stiffness, damping, effort, velocity, friction=.65):
    interface = PositionPDInterface(limits, stiffness, damping, effort, velocity, friction)
    return LandingSupportMPCV1(interface)
