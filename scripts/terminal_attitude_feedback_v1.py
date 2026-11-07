"""Small support-kinematic attitude corrections to the terminal position target.

This candidate does not invent stance forces or change physical motor gains.
Its sign comes from J_leg*dq + J_root*dtheta = 0 at measured support feet.
Forward dynamics and native execution must separately validate its utility.
"""
import math
import numpy as np
import pinocchio as pin


class TerminalAttitudeFeedbackV1:
    def __init__(self, base, proportional=.15, damping_s=.02, maximum_joint_delta_rad=.04,
                 regularization_m=.003, ramp_s=.2):
        values = [proportional, damping_s, maximum_joint_delta_rad, regularization_m, ramp_s]
        if not np.isfinite(values).all() or not (0 < proportional <= .5 and 0 <= damping_s <= .1
                and 0 < maximum_joint_delta_rad <= .08 and regularization_m > 0 and ramp_s > 0):
            raise ValueError('Explicit bounded attitude correction parameters required')
        self.base = base
        self.model = base.model
        self.interface = base.interface
        self.proportional = proportional
        self.damping_s = damping_s
        self.maximum_joint_delta_rad = maximum_joint_delta_rad
        self.regularization_m = regularization_m
        self.ramp_s = ramp_s
        self.last_diagnostic = None

    def action(self, offset, state, previous):
        target, mode = self.base.action(offset, state, previous)
        self.last_diagnostic = dict(active=False, offset=offset)
        if offset < 80 or not np.asarray(state.contacts, bool).any():
            return target, mode
        rotation = state.root_rotation_world
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        upright = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        error_world = pin.log3(upright @ rotation.T)
        omega_world = rotation @ state.v[3:6]
        desired_world = self.proportional*error_world - self.damping_s*omega_world
        # Keep heading control with the frozen policy; only right the body.
        desired_world[2] = 0.
        blend = np.clip((offset-80)*.01/self.ramp_s, 0., 1.)
        desired_local = rotation.T @ (blend*desired_world)
        snapshot = self.model.snapshot(state.q, state.v)
        contacts = np.asarray(state.contacts, bool)
        jacobian = snapshot['foot_jacobian'][contacts].reshape(-1, self.model.model.nv)
        leg = jacobian[:, self.model.v_indices[:12]]
        rhs = -(jacobian[:, 3:6] @ desired_local)
        # Damped least squares gives the minimum-norm joint correction even
        # with only one or two current contacts. No future foot is latched in.
        delta = leg.T @ np.linalg.solve(leg @ leg.T + self.regularization_m**2*np.eye(len(rhs)), rhs)
        delta = np.clip(delta, -self.maximum_joint_delta_rad, self.maximum_joint_delta_rad)
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite support-kinematic correction')
        proposed = target.copy()
        proposed[:12] += delta
        proposed[:12] = np.clip(proposed[:12], self.interface.limits[:12, 0], self.interface.limits[:12, 1])
        realized_delta = proposed[:12] - target[:12]
        self.last_diagnostic = dict(active=True, offset=offset, current_contact_indices=np.flatnonzero(contacts).tolist(),
            blend=float(blend), error_world_rad=error_world.tolist(),
            desired_virtual_root_rotation_local_rad=desired_local.tolist(),
            applied_joint_target_delta_rad=realized_delta.tolist(),
            virtual_foot_shift_without_joint_correction_m=float(np.linalg.norm(rhs)),
            virtual_linearized_foot_shift_after_correction_m=float(np.linalg.norm(leg @ realized_delta-rhs)),
            scope='Virtual kinematic consistency, not a predicted or measured physical foot tracking error')
        return proposed, 'terminal_support_with_attitude_feedback_candidate'
