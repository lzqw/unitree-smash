"""Try sustained support once the current state is ready, before a plan ends.

Uses the unchanged instantaneous stability/geometry checks as eligibility,
not as success. The reference can be an incomplete offline planning candidate;
this controller and all resulting trajectories still need complete execution.
One constrained full18 solve owns the output. No contacts or forces are saved
as facts, and the final 0.8-second/end-of-episode criterion is unchanged.
"""
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15


class PureControlStateSettlingV23(PureControlStanceFootReferenceV15):
    def _reference_time(self):
        if self.terminal_stance is not None:
            return None, 0.
        return super()._reference_time()

    def command(self, state, phase, time_s):
        captured = False
        conditions = {}
        if (self.terminal_stance is None and self.started is not None
                and self.chosen is not None):
            eligible, conditions = self.terminal_candidate(state)
            if eligible:
                snapshot = self.model.snapshot(state.q, state.v)
                self.terminal_stance = dict(com=snapshot['com_world'].copy(),
                    rotation=state.root_rotation_world.copy(),
                    joint=state.joint_position.copy(),
                    captured_tick=int(round((float(time_s)-self.started)/.01)))
                self.nominal = state.joint_position.copy()
                self.initial_arm = state.joint_position[12:].copy()
                captured = True
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_state_settling_v23',
            state_settling_capture_now=captured,
            state_settling_conditions=conditions,
            settling_requires_reference_end=False,
            full_episode_certified=False)
        return target, diagnostic
