"""Optimize early reference timing without changing the physical episode clock."""
import copy

import numpy as np
from scipy.interpolate import PchipInterpolator

from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference
from scripts.search_landing_support_tail_v3 import SupportTailReference
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.current_state_recovery_planner_v3 import build_feedback_plan


class TimedEarlyReferenceV1(WholeBodyEarlyReference):
    def __init__(self, model, interface, initial, knots, timing):
        times = np.asarray(timing, float)
        if (times.shape != (4,) or not np.isfinite(times).all()
                or not .02 <= times[0] <= .12
                or not times[0]+.02 <= times[1] <= .30
                or not times[1]+.03 <= times[2] <= .75
                or not .02 <= times[3] <= .15):
            raise ValueError('Ordered bounded reference times and braking duration required')
        super().__init__(model, interface, initial, knots)
        self.timing = times.copy()
        self.leg_curve = PchipInterpolator(np.r_[0., times[:3]], np.vstack([
            initial.previous_command_target[:12], np.clip(knots[:,:12], interface.limits[:12,0], interface.limits[:12,1])]), axis=0)
        self.arm_curve = PchipInterpolator(np.r_[0., times[:3]], np.vstack([np.zeros(6), knots[:,12:]]), axis=0)

    def action(self, offset, state, previous):
        step = offset-80
        if type(step) is not int or not 0 <= step < 520:
            raise ValueError('Original recovery clock required')
        t = .01*step
        u = float(np.clip(t/self.timing[3], 0., 1.)); blend = u*u*(3.-2.*u)
        target = self.initial.previous_command_target.copy()
        target[:12] = self.leg_curve(min(t, self.timing[2]))
        target[12:] = ((1.-blend)*target[12:]+blend*self.initial.joint_position[12:]
            +self.arm_curve(min(t, self.timing[2])))
        target = np.clip(target, self.interface.limits[:,0], self.interface.limits[:,1])
        target[12:] = np.clip(target[12:], previous[12:]-.01*self.interface.arm_rate,
            previous[12:]+.01*self.interface.arm_rate)
        return target, 'timed_continuous_whole_body_reference'


def timed_factory(model, interface, initial, knots, *, reference_q, reference_v, gains, timing):
    knots = np.asarray(knots, float)
    base = SupportTailReference(model, interface, initial, knots[3:], fixed_knots=knots[:3])
    early = TimedEarlyReferenceV1(model, interface, initial, knots[:3], timing)
    # SupportTailReference reconstructs its raw target after 0.8 seconds.
    # Its historical calls cap at 0.5 seconds: by 0.8 the early trajectory
    # must instead have reached the new last knot, even if that knot is later.
    early.legs = lambda ignored_t: early.leg_curve(early.timing[2])
    early.arms = lambda ignored_t: early.arm_curve(early.timing[2])
    base.base = early
    return LandingReferenceTrackingV1(base, reference_q, reference_v, **gains)


def build_timed_plan(worker, plan):
    if plan.get('contract') != 'timed_foot_space_recovery_v1':
        raise ValueError('Explicit timed-reference contract required')
    original_factory = worker['factory']
    keys = original_factory.keywords
    if set(keys) != {'reference_q', 'reference_v', 'gains'}:
        raise ValueError('Declared reference-tracking factory required')
    def factory(model, interface, initial, knots):
        return timed_factory(model, interface, initial, knots, **keys, timing=plan['early_timing_s'])
    local = dict(worker, factory=factory)
    # Reuse the unchanged attitude, transition, braking and correction layers.
    # The old standalone loader cannot accept the new contract by accident.
    underlying = copy.deepcopy(plan)
    underlying['contract'] = 'foot_space_current_state_plan_v1'
    return build_feedback_plan(local, underlying)
