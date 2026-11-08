"""Experimental terminal-stance feedback after an optimized landing path.

Capture the actually reached stance, not a command frame or future outcome.
Keep solving the same constrained whole-body problem with live contacts. This
tests the generic standing-reference transition; it is not a safety guarantee,
an invariant terminal set, or a change to physical/stability requirements.
"""
import numpy as np
import pinocchio as pin

from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12


class PureControlTerminalStanceV13(PureControlReferenceClockV12):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_stance = None

    def command(self, state, phase, time_s):
        # No prefix mutation: only a finished, actually selected reference
        # with observed support can establish this experimental soft target.
        if (self.terminal_stance is None and self.started is not None
                and self.chosen is not None and state.contacts.any()):
            tick = int(round((float(time_s) - self.started) / .01))
            if tick >= len(self.chosen['trajectory']):
                snap = self.model.snapshot(state.q, state.v)
                self.terminal_stance = dict(
                    com=snap['com_world'].copy(),
                    rotation=state.root_rotation_world.copy(),
                    joint=state.joint_position.copy(), captured_tick=tick)
                # This is a posture objective, never an executable hold.
                # It also keeps an eventual uncertified fallback consistent
                # with the new reference instead of the old nominal pose.
                self.nominal = state.joint_position.copy()
                self.initial_arm = state.joint_position[12:].copy()
        target, diagnostic = super().command(state, phase, time_s)
        diagnostic.update(controller='pure_control_terminal_stance_v13',
            terminal_stance_active=self.terminal_stance is not None,
            terminal_stance_certified=False,
            terminal_stance_captured_tick=(self.terminal_stance['captured_tick']
                if self.terminal_stance is not None else None))
        return target, diagnostic

    def reference(self, state, snapshot, joint_reference):
        result = super().reference(state, snapshot, joint_reference)
        if self.terminal_stance is None:
            return result
        terminal = self.terminal_stance
        cfg = self.settings
        if state.contacts.any():
            result['com_position'] = terminal['com'].copy()
            result['com_acceleration'] = (
                cfg.com_kp_s2 * (terminal['com'] - snapshot['com_world'])
                - cfg.com_kd_s * snapshot['com_velocity_world'])
        # In flight the inherited reference retains gravity; no ground
        # contacts are created or latched by this controller.
        result['angular_acceleration_body'] = (
            cfg.orientation_kp_s2 * pin.log3(
                state.root_rotation_world.T @ terminal['rotation'])
            - cfg.orientation_kd_s * state.v[3:6])
        result['joint_acceleration'] = (
            cfg.posture_kp_s2 * (terminal['joint'] - state.joint_position)
            - cfg.posture_kd_s * state.joint_velocity)
        self.reference_diagnostics.update(
            terminal_stance_soft_target=True,
            com_goal=result['com_position'].copy(),
            original_physical_constraints=True)
        return result
