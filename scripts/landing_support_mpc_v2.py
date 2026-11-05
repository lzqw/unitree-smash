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
import math

import numpy as np

from scripts.landing_support_mpc_v1 import LandingSupportMPCV1, LandingMPCSettingsV1


@dataclass(frozen=True)
class LandingMPCSettingsV2(LandingMPCSettingsV1):
    support_margin_trigger_m: float = .025
    terminal_support_margin_m: float = .010
    two_foot_max_com_distance_m: float = .075
    one_foot_max_com_distance_m: float = .045


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
        if settings.terminal_support_margin_m > settings.support_margin_trigger_m:
            raise ValueError('terminal support margin must not exceed trigger margin')
        super().__init__(interface, qp_settings=qp_settings, settings=settings)

    def _support_geometry(self, state):
        snapshot = self.qp.model.snapshot(state.q, state.v)
        feet_ids = [self.qp.model.names.index(name) for name in self.qp.model.feet]
        feet = snapshot['body_positions_world'][feet_ids]
        active = np.asarray(state.contacts, dtype=bool)
        return snapshot, feet, feet[active]

    def support_margin(self, state):
        snapshot, _, active_feet = self._support_geometry(state)
        return _support_margin(snapshot['com_world'], active_feet)

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
        result = super().solve(state, joint_reference, force=force)
        if not result.get('horizon'):
            return result
        # V1's terminal risk was computed with its own risk method (dynamic
        # dispatch already gives V2 support margin), but its terminal gate did
        # not include the support condition.  Recompute from the predicted
        # terminal state by replaying the accepted accelerations.
        predicted = state
        for row in result['horizon']:
            if 'acceleration' not in row:
                break
            predicted = self._integrate(predicted, np.asarray(row['acceleration']))
        terminal = self.risk(predicted)
        result['terminal_risk'] = terminal
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
