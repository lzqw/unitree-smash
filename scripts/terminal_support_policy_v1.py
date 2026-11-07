"""Nominal model trajectory with leg feedback and a persistent terminal controller.

This is a candidate local controller for MPC's terminal phase, not online
replanning or a certified invariant set. Motor gains and limits stay unchanged.
"""
import numpy as np


class TerminalSupportPolicyV1:
    def __init__(self, model, interface, initial, warm, terminal):
        final = warm['final']
        if not final['accepted'] or warm['horizon_steps'] != 80:
            raise ValueError('Complete model-accepted prefix required')
        if (terminal['failure'] is not None or terminal['valid_steps'] != 260 or
                terminal['model_window']['longest_model_kinematic_window_s'] < .8 or
                terminal['mode'] != 'fixed_preload_leg_feedback_and_hip_reference'):
            raise ValueError('Require a complete model-verified feedback tail')
        self.commands = np.asarray(final['commands'], float)
        if self.commands.shape != (80,18) or len(final['states']) != 80:
            raise ValueError('Incomplete prefix')
        self.model = model; self.interface = interface
        self.q_reference = np.vstack([initial.joint_position,
            [np.asarray(s['q'])[model.q_indices] for s in final['states']]])
        self.v_reference = np.vstack([initial.joint_velocity,
            [np.asarray(s['v'])[model.v_indices] for s in final['states']]])
        self.preload = np.asarray(terminal['initial_static_reference']['target'], float)
        self.kp_feedback = float(terminal['posture_gain'])
        self.kd_feedback = float(terminal['velocity_gain'])
        self.hip_bias = float(terminal['hip_bias_rad'])
        if (self.preload.shape != (18,) or not np.isfinite(self.preload).all() or
                self.kp_feedback != 2. or self.kd_feedback != .5 or abs(self.hip_bias) > .18):
            raise ValueError('Undeclared terminal feedback parameters')

    def action(self, offset, state, previous):
        if type(offset) is not int or not 0 <= offset < 340:
            raise ValueError('Original remaining episode control clock required')
        if offset < 80:
            request = self.commands[offset].copy()
            qref = self.q_reference[offset]; vref = self.v_reference[offset]
            mode = 'nominal_prefix_with_leg_feedback'
        else:
            request = self.preload.copy(); request[:4] += self.hip_bias
            qref = self.q_reference[-1]; vref = np.zeros(18)
            mode = 'persistent_terminal_leg_feedback'
        request[:12] += self.kp_feedback*(qref[:12]-state.joint_position[:12])
        request[:12] -= self.kd_feedback*(state.joint_velocity[:12]-vref[:12])/self.interface.kp[:12]
        if not np.isfinite(request).all(): raise ValueError('Nonfinite feedback request')
        target = np.clip(request, self.interface.limits[:,0], self.interface.limits[:,1])
        target[12:] = np.clip(target[12:], np.asarray(previous)[12:]-.01*self.interface.arm_rate,
                            np.asarray(previous)[12:]+.01*self.interface.arm_rate)
        return target, mode
