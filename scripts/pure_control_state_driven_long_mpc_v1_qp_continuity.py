"""Long-horizon MPC with command continuity solved inside the interval QP."""
from dataclasses import replace

from scripts.pure_control_state_driven_long_mpc_v1 import PureControlStateDrivenLongMPCV1


class PureControlStateDrivenLongMPCV1QPContinuity(PureControlStateDrivenLongMPCV1):
    """Keep v1 timing and horizon; add a native-target continuity cost."""

    command_continuity_weight = 1.0

    def __init__(self, interface, nominal_posture):
        super().__init__(interface, nominal_posture)
        self.settings = replace(
            self.settings,
            command_continuity_weight=float(self.command_continuity_weight),
        )

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic = dict(diagnostic)
        diagnostic.update(
            controller='pure_control_state_driven_long_mpc_v1_qp_continuity',
            command_continuity_weight=float(self.command_continuity_weight),
            post_solve_command_blend=False,
        )
        return target, diagnostic


__all__ = ['PureControlStateDrivenLongMPCV1QPContinuity']
