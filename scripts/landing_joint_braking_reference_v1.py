"""Causal joint-limit braking candidate through original position targets.

This feedback is not a barrier certificate: approximate stopping distance
only selects a continuous correction. Full coupled rollout decides feasibility.
"""
import numpy as np


class LandingJointBrakingReferenceV1:
    def __init__(self, base, *, gain_s, lookahead_s, margin_rad=.15, maximum_delta_rad=.8):
        values = np.asarray([gain_s, lookahead_s, margin_rad, maximum_delta_rad], float)
        if (not np.isfinite(values).all() or not 0 <= gain_s <= .3 or
                not 0 <= lookahead_s <= .2 or not 0 < margin_rad <= .5 or
                not 0 < maximum_delta_rad <= 1.):
            raise ValueError('Finite bounded braking candidate required')
        self.base = base; self.model = base.model; self.interface = base.interface
        self.gain_s = gain_s; self.lookahead_s = lookahead_s
        self.margin_rad = margin_rad; self.maximum_delta_rad = maximum_delta_rad

    def action(self, offset, state, previous):
        target, mode = self.base.action(offset, state, previous)
        if offset == 80 or self.gain_s == 0:
            return target, mode
        q = state.joint_position[:12]; v = state.joint_velocity[:12]
        limits = self.interface.limits[:12]
        distance = np.where(v > 0, limits[:, 1] - q, q - limits[:, 0])
        weight = np.clip((self.margin_rad + self.lookahead_s * np.abs(v) - distance) / self.margin_rad, 0., 1.)
        weight = weight * weight * (3. - 2. * weight)
        delta = np.clip(-self.gain_s * v * weight, -self.maximum_delta_rad, self.maximum_delta_rad)
        result = target.copy()
        result[:12] = np.clip(result[:12] + delta, limits[:, 0], limits[:, 1])
        return result, 'causal_joint_limit_braking_candidate'
