#!/usr/bin/env python3
"""Support-margin-aware landing MPC built on the verified support QP.

The v1 controller triggers on tilt/speed/support-count only.  The native
landing audit showed a different failure mode: feet make contact, then the
COM drifts outside the available support region and a body link touches the
ground later.  This module adds a geometric support-margin signal and a
terminal viability check while keeping the original QP and position-target
interface unchanged.

This is still a model-level controller.  It does not create contacts, model
impact impulses, or turn a position-PD interface into a torque WBC.
"""
from dataclasses import dataclass, asdict
import copy
import math

import numpy as np

from scripts.landing_support_mpc_v1 import LandingSupportMPCV1, LandingMPCSettingsV1


@dataclass(frozen=True)
class LandingMPCSettingsV2(LandingMPCSettingsV1):
    support_margin_trigger_m: float = .025
    terminal_support_margin_m: float = .010
    two_foot_max_com_distance_m: float = .075
    one_foot_max_com_distance_m: float = .045
    landing_leg_posture_blend: float = .35
    landing_arm_posture_blend: float = 0.
    predict_touchdown_contacts: bool = False
    touchdown_height_tolerance_m: float = .012
    touchdown_vertical_speed_limit_mps: float = 1.5


def _hull(points):
    """Monotonic-chain convex hull for finite 2-D points."""
    points = sorted({(float(x), float(y)) for x, y in np.asarray(points)})
    if len(points) <= 1:
        return np.asarray(points, dtype=np.float64)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _segment_distance(point, a, b):
    d = b - a
    den = float(d @ d)
    if den <= 1e-12:
        return float(np.linalg.norm(point-a))
    u = float(np.clip((point-a) @ d / den, 0., 1.))
    return float(np.linalg.norm(point-(a+u*d)))


def _support_margin(point, feet):
    """Signed horizontal support margin in metres.

    Positive means the COM projection is inside a 3+ foot convex hull.  For
    one/two contacts, the support set is degenerate; the returned negative
    distance expresses how far the COM is from the available contact set.
    """
    point = np.asarray(point, dtype=np.float64)[:2]
    feet = np.asarray(feet, dtype=np.float64)[:, :2]
    if len(feet) >= 3:
        hull = _hull(feet)
        if len(hull) >= 3:
            margins = []
            for i, a in enumerate(hull):
                b = hull[(i+1) % len(hull)]
                edge = b-a
                length = float(np.linalg.norm(edge))
                if length > 1e-12:
                    margins.append(float(np.cross(edge, point-a)/length))
            return float(min(margins)) if margins else -np.inf
    if len(feet) == 2:
        return -_segment_distance(point, feet[0], feet[1])
    if len(feet) == 1:
        return -float(np.linalg.norm(point-feet[0]))
    return -np.inf


class LandingSupportMPCV2(LandingSupportMPCV1):
    """V1 fixed-contact support MPC with COM/support-margin viability gates."""

    def __init__(self, interface, *, qp_settings=None,
                 settings=LandingMPCSettingsV2()):
        if not isinstance(settings, LandingMPCSettingsV2):
            raise ValueError('LandingSupportMPCV2 requires LandingMPCSettingsV2')
        for name in ('support_margin_trigger_m', 'terminal_support_margin_m',
                     'two_foot_max_com_distance_m', 'one_foot_max_com_distance_m'):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('landing_leg_posture_blend', 'landing_arm_posture_blend'):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0. <= value <= 1.:
                raise ValueError(f'{name} must be in [0,1]')
        if not isinstance(settings.predict_touchdown_contacts, bool):
            raise ValueError('predict_touchdown_contacts must be boolean')
        for name in ('touchdown_height_tolerance_m', 'touchdown_vertical_speed_limit_mps'):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if settings.terminal_support_margin_m > settings.support_margin_trigger_m:
            raise ValueError('terminal support margin must not exceed trigger margin')
        super().__init__(interface, qp_settings=qp_settings, settings=settings)
        self.posture_target = None

    def set_posture_target(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (18,) or not np.isfinite(target).all():
            raise ValueError('posture target must be a finite 18-vector')
        self.posture_target = target.copy()

    def _support_geometry(self, state):
        snapshot = self.qp.model.snapshot(state.q, state.v)
        feet_ids = [self.qp.model.names.index(name) for name in self.qp.model.feet]
        feet = snapshot['body_positions_world'][feet_ids]
        active = np.asarray(state.contacts, dtype=bool)
        return snapshot, feet, feet[active]

    def support_margin(self, state):
        snapshot, _, active_feet = self._support_geometry(state)
        return _support_margin(snapshot['com_world'], active_feet)

    def _after_prediction(self, state):
        """Approximate imminent touchdown for the next QP stage.

        This is intentionally an event approximation, not an impact model. A
        new contact is added only when an existing support foot defines a ground
        height and an inactive foot is close to that height while moving
        downward. The following QP stage can then allocate reaction force to
        that foot instead of keeping the old contact mode for the whole
        horizon. Actual simulator contact remains authoritative at the next
        control tick.
        """
        if not self.settings.predict_touchdown_contacts:
            return state, []
        active = np.asarray(state.contacts, dtype=bool)
        if not active.any():
            return state, []
        snapshot = self.qp.model.snapshot(state.q, state.v)
        feet_ids = [self.qp.model.names.index(name) for name in self.qp.model.feet]
        feet = snapshot['body_positions_world'][feet_ids]
        foot_velocity = snapshot['foot_jacobian'] @ state.v
        ground_z = float(np.median(feet[active, 2]))
        candidates = []
        for index in np.flatnonzero(~active):
            height = float(feet[index, 2] - ground_z)
            vertical_speed = float(foot_velocity[index, 2])
            # World Z is positive upward.  A predicted touchdown must be
            # descending (or momentarily stationary), while excluding a foot
            # moving downward faster than the configured impact limit.
            if (height <= self.settings.touchdown_height_tolerance_m and
                    -self.settings.touchdown_vertical_speed_limit_mps <= vertical_speed <= 0.0):
                candidates.append(int(index))
        if not candidates:
            return state, []
        next_state = copy.deepcopy(state)
        next_state.contacts[candidates] = True
        return next_state, candidates

    def risk(self, state):
        result = super().risk(state)
        margin = self.support_margin(state)
        result['support_margin_m'] = margin
        # A negative/low margin is an early indicator even when tilt and speed
        # are still below the v1 thresholds.
        if margin < self.settings.support_margin_trigger_m:
            result['trigger'] = True
            result['reason'] = 'support_margin'
        return result

    def risk_score(self, risk):
        base = super().risk_score(risk)
        margin = float(risk.get('support_margin_m', 0.))
        deficit = max(0., self.settings.support_margin_trigger_m - margin)
        return base + deficit / self.settings.support_margin_trigger_m

    def _terminal_ok(self, state, risk):
        margin = float(risk.get('support_margin_m', -np.inf))
        feet = int(np.asarray(state.contacts, dtype=bool).sum())
        if feet >= 3:
            support_ok = margin >= self.settings.terminal_support_margin_m
        elif feet == 2:
            # Two-foot support has zero polygon area; accept only a small COM
            # distance to the line, with the original velocity gates intact.
            support_ok = margin >= -self.settings.two_foot_max_com_distance_m
        elif feet == 1:
            support_ok = margin >= -self.settings.one_foot_max_com_distance_m
        else:
            support_ok = False
        return bool(support_ok and
                    risk['tilt_rad'] <= self.settings.terminal_tilt_rad and
                    risk['angular_speed_rad_s'] <= self.settings.terminal_angular_speed_rad_s and
                    risk['horizontal_speed_mps'] <= self.settings.terminal_horizontal_speed_mps)

    def solve(self, state, joint_reference, *, force=False):
        # Copy V1 implementation's rollout but apply the support-margin risk
        # and terminal viability checks.  Keeping the same QP preserves the
        # established solver diagnostics and action interface.
        effective_reference = np.asarray(joint_reference, dtype=np.float64).copy()
        if self.posture_target is not None and int(np.asarray(state.contacts, dtype=bool).sum()) > 0:
            weights = np.r_[
                np.full(12, self.settings.landing_leg_posture_blend),
                np.full(6, self.settings.landing_arm_posture_blend)]
            effective_reference = ((1. - weights) * effective_reference
                                   + weights * self.posture_target)
        result = super().solve(state, effective_reference, force=force)
        result['posture_reference_used'] = effective_reference.tolist()
        if not result.get('horizon'):
            return result
        # V1's terminal risk was computed with its own risk method (dynamic
        # dispatch already gives V2 support margin), but its terminal gate did
        # not include the support condition.  Reuse the exact same replay as
        # the rollout: executor realization and predicted contact transitions
        # must not silently disappear during terminal validation.
        predicted, replayed_contact_additions = self.replay_horizon(state, result['horizon'])
        terminal = self.risk(predicted)
        result['terminal_risk'] = terminal
        result['terminal_replayed_contact_additions'] = replayed_contact_additions
        result['support_margin_terminal_ok'] = self._terminal_ok(predicted, terminal)
        if result.get('accepted') and not result['support_margin_terminal_ok']:
            result['accepted'] = False
            result['rejection'] = 'terminal support margin not reached'
        return result

    def contract(self):
        result = super().contract()
        result['schema'] = 'landing-support-mpc-v2-support-margin'
        result['formulation'] = 'fixed-contact-sequential-QP-rollout-with-support-margin-gate'
        result['settings'] = asdict(self.settings)
        result['solver_failure_is_not_hardware_impossibility'] = True
        return result


__all__ = ['LandingMPCSettingsV2', 'LandingSupportMPCV2', '_support_margin']
