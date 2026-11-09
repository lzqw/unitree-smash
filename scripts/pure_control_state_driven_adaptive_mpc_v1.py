"""Phase-adaptive state-driven landing MPC.

The 0.32 s controller was effective, while a fixed 0.64 s horizon accumulated
too much contact-model error.  This variant keeps a longer look-ahead before
touchdown but shortens the horizon after contact is established:
AIR_PREPARE=0.40 s, BUFFER_LOAD=0.24 s, SUSTAIN=0.12 s.  All horizons are
receding and state-driven; no reference-library lookup is used.
"""
from scripts.pure_control_state_driven_long_mpc_v1 import PureControlStateDrivenLongMPCV1


class PureControlStateDrivenAdaptiveMPCV1(PureControlStateDrivenLongMPCV1):
    phase_horizon_steps = {
        'AIR_PREPARE': 40,
        'BUFFER_LOAD': 24,
        'SUSTAIN': 12,
    }
    phase_replan_stride = {
        'AIR_PREPARE': 8,
        'BUFFER_LOAD': 8,
        'SUSTAIN': 8,
    }

    def command(self, state, phase, time_s):
        self.horizon_steps = self.phase_horizon_steps[phase]
        self.replan_stride = self.phase_replan_stride[phase]
        return super().command(state, phase, time_s)


__all__ = ['PureControlStateDrivenAdaptiveMPCV1']
