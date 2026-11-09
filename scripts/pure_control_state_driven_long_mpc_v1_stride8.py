"""Fixed 0.32 s state-driven long-horizon MPC, replanning every 8 ticks."""
from scripts.pure_control_state_driven_long_mpc_v1 import PureControlStateDrivenLongMPCV1

class PureControlStateDrivenLongMPCV1Stride8(PureControlStateDrivenLongMPCV1):
    replan_stride = 8

    def command(self, state, phase, time_s):
        target, diag = super().command(state, phase, time_s)
        diag = dict(diag)
        diag['controller'] = 'pure_control_state_driven_long_mpc_v1_stride8'
        diag['replan_stride'] = self.replan_stride
        return target, diag

__all__ = ['PureControlStateDrivenLongMPCV1Stride8']
