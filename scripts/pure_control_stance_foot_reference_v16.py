"""V15 with a command-continuous fallback after an unsuccessful interval QP.

The V15 fallback rebuilt a posture target from the nominal reference and
preload when the interval solver was infeasible.  Native audits showed that
this uncertified fallback was the last command before most head contacts.
V16 keeps the last admitted native target instead, applying only the same
one-tick velocity admission and a small measured-velocity damping term.  It
does not claim that a held command is safe; it removes an unbounded posture
jump so the next tick can replan from the actual state.
"""
import numpy as np

from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15


class PureControlStanceFootReferenceV16(PureControlStanceFootReferenceV15):
    """Command-continuous recovery candidate; physical limits are unchanged."""

    def command(self, state, phase, time_s):
        target, diagnostic = super().command(state, phase, time_s)
        if diagnostic.get('source') != 'uncertified_posture_fallback':
            diagnostic.update(controller='pure_control_stance_foot_reference_v16',
                              fallback_mode='unused_interval_qp_target')
            return target, diagnostic

        # The previous target is the last command that passed through the
        # native position interface.  Re-admit it against the current state
        # rather than sending the nominal posture from V1.  This is a
        # continuity fallback, not a stability certificate.
        self.interface.bind_previous_command(state.previous_command_target)
        low, high = self.interface.admitted_target_bounds(
            state.joint_position, .01)
        previous = np.asarray(state.previous_command_target, dtype=np.float64)
        velocity = np.asarray(state.joint_velocity, dtype=np.float64)
        if previous.shape != (18,) or velocity.shape != (18,):
            raise ValueError('Finite 18D previous target and velocity required')
        damped = previous - .01 * np.clip(velocity, -5., 5.)
        target = np.clip(damped, low, high)
        diagnostic.update(
            controller='pure_control_stance_foot_reference_v16',
            source='command_continuous_fallback',
            fallback_mode='hold_last_admitted_target_with_velocity_damping',
            fallback_certified=False,
            fallback_replaced_nominal_posture=True,
            fallback_target_jump_rad=float(np.max(np.abs(target - previous))),
        )
        return target, diagnostic


__all__ = ['PureControlStanceFootReferenceV16']
