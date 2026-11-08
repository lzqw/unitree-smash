"""Choose a terminal stance from current settling evidence, not horizon end.

An optimized path may end during rebound. Keep the existing continuous
recovery controller until its current state meets the original instantaneous
stability bounds. That is only permission to try a soft stance objective;
success still requires the full original episode and physical collision audit.
"""
import numpy as np

from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.pure_control_terminal_stance_v13 import PureControlTerminalStanceV13


class PureControlSettledStanceV14(PureControlTerminalStanceV13):
    def terminal_candidate(self, state):
        r = state.root_rotation_world
        angles = np.array([np.arctan2(r[2, 1], r[2, 2]),
            np.arcsin(np.clip(-r[2, 0], -1., 1.))])
        velocity = r @ state.v[:3]
        conditions = dict(
            measured_support=bool(state.contacts.any()),
            attitude=bool(np.max(abs(angles)) <= np.deg2rad(15.)),
            horizontal_speed=bool(np.linalg.norm(velocity[:2]) <= .3),
            vertical_speed=bool(abs(velocity[2]) <= .1),
            angular_speed=bool(np.linalg.norm(state.v[3:6]) <= 1.),
            joint_speed=bool(np.max(abs(state.joint_velocity)) <= 1.))
        if all(conditions.values()):
            gaps, _ = self.geometry.evaluate(state.q)
            conditions['authored_ground_clearance'] = bool(np.min(gaps) >= 0.)
        return all(conditions.values()), conditions

    def command(self, state, phase, time_s):
        eligible = False
        conditions = {}
        if (self.terminal_stance is None and self.started is not None
                and self.chosen is not None):
            tick = int(round((float(time_s) - self.started) / .01))
            if tick >= len(self.chosen['trajectory']):
                eligible, conditions = self.terminal_candidate(state)
                if eligible:
                    snap = self.model.snapshot(state.q, state.v)
                    self.terminal_stance = dict(
                        com=snap['com_world'].copy(),
                        rotation=state.root_rotation_world.copy(),
                        joint=state.joint_position.copy(), captured_tick=tick)
                    self.nominal = state.joint_position.copy()
                    self.initial_arm = state.joint_position[12:].copy()
        # Deliberately skip V13's unconditional horizon-end capture. Its
        # reference method is retained; the full constrained solve is intact.
        target, diagnostic = PureControlReferenceClockV12.command(
            self, state, phase, time_s)
        diagnostic.update(controller='pure_control_settled_stance_v14',
            terminal_stance_active=self.terminal_stance is not None,
            terminal_stance_certified=False, terminal_candidate_now=eligible,
            terminal_candidate_conditions=conditions,
            terminal_stance_captured_tick=(self.terminal_stance['captured_tick']
                if self.terminal_stance is not None else None))
        return target, diagnostic
