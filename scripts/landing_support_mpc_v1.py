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

    def _integrate(self, state, acceleration):
        next_state = copy.deepcopy(state)
        dt = self.settings.control_dt_s
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

    def solve(self, state, joint_reference, *, force=False):
        started = __import__('time').perf_counter()
        risk = self.risk(state)
        if not force and not risk['trigger']:
            return dict(accepted=False, intervened=False, risk=risk,
                        rejection='risk trigger not active', horizon=[],
                        native_execution=False, mpc_contract=self.contract())
        predicted = copy.deepcopy(state)
        horizon = []
        for stage in range(self.settings.horizon_steps):
            plan = self.qp.solve(predicted, joint_reference)
            row = dict(stage=stage, accepted=bool(plan.get('accepted')), solver_status=plan.get('solver_status'),
                       solver_iterations=plan.get('solver_iterations'), wall_runtime_s=plan.get('wall_runtime_s'),
                       contacts=predicted.contacts.tolist(), risk=self.risk(predicted))
            if not plan.get('accepted'):
                return dict(accepted=False, intervened=True, risk=risk, horizon=horizon + [row],
                            rejection=f'QP rejected prediction stage {stage}', native_execution=False,
                            solver_failure=True, wall_runtime_s=__import__('time').perf_counter()-started,
                            mpc_contract=self.contract())
            horizon.append(dict(row, residuals=plan.get('residuals'),
                                normalized_action=plan['normalized_action'].tolist(),
                                joint_position_target=plan['joint_position_target'].tolist()))
            predicted = self._integrate(predicted, plan['acceleration'])
        terminal = self.risk(predicted)
        terminal_ok = bool(terminal['tilt_rad'] <= self.settings.terminal_tilt_rad
                           and terminal['angular_speed_rad_s'] <= self.settings.terminal_angular_speed_rad_s
                           and terminal['horizontal_speed_mps'] <= self.settings.terminal_horizontal_speed_mps)
        if not terminal_ok:
            return dict(accepted=False, intervened=True, risk=risk, horizon=horizon,
                        terminal_risk=terminal, rejection='terminal support margin not reached',
                        solver_failure=False, native_execution=False,
                        wall_runtime_s=__import__('time').perf_counter()-started,
                        mpc_contract=self.contract())
        first = horizon[0]
        return dict(accepted=True, intervened=True, risk=risk, horizon=horizon,
                    terminal_risk=terminal, normalized_action=np.asarray(first['normalized_action']),
                    joint_position_target=np.asarray(first['joint_position_target']),
                    solver_failure=False, native_execution=False,
                    wall_runtime_s=__import__('time').perf_counter()-started,
                    mpc_contract=self.contract())


def make_default_landing_mpc(limits, stiffness, damping, effort, velocity, friction=.65):
    interface = PositionPDInterface(limits, stiffness, damping, effort, velocity, friction)
    return LandingSupportMPCV1(interface)
