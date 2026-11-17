"""Loaded support QP with a backward-Euler position-drive command map.

Candidate, not a PhysX replica. The QP and its one-step state use the same
discretization. Original effort, velocity, position, friction and command-history
bounds remain hard. Saturated drive branches, impacts and sliding are not
modelled here: this QP searches the unsaturated implicit-drive branch only.
"""
import time
import numpy as np
import osqp
from scipy import sparse

from scripts.command_history_support_qp_v1 import CommandHistorySupportQP
from scripts.frame_point_kinematics_v1 import frame_point_kinematics


def implicit_command_map(torque_map, torque_bias, q, v, kp, kd, selector, dt):
    """u = q+dt*v_next+(tau+kd*v_next)/kp, v_next=v+dt*a."""
    return (torque_map / kp[:, None] +
            (dt * (dt + kd / kp))[:, None] * selector,
            q + (dt + kd / kp) * v + torque_bias / kp)


class ImplicitCommandSupportQP(CommandHistorySupportQP):
    def angular_task(self, state, snapshot, reference):
        """Default body-angular acceleration task; subclasses may supply a task."""
        matrix = np.zeros((3, 24)); matrix[:, 3:6] = np.eye(3)
        return matrix, reference['angular_acceleration_body']

    def solve(self, state, joint_reference, *, static_equilibrium=False):
        started = time.perf_counter()
        cfg, interface = self.settings, self.interface
        if cfg.force_distribution_weight or cfg.force_wrench_weight or cfg.nonfoot_clearance_hard:
            raise ValueError('This candidate does not silently omit optional force/hard-clearance tasks')
        dt = cfg.control_dt_s
        interface.bind_previous_command(state.previous_command_target)
        requested = np.asarray(joint_reference, dtype=float)
        if (requested.shape != (18,) or not np.isfinite(requested).all() or
                np.any(requested < interface.limits[:, 0] - 1e-6) or
                np.any(requested > interface.limits[:, 1] + 1e-6)):
            raise ValueError('Finite reference inside original joint bounds required')
        if static_equilibrium and np.max(abs(state.v)) > 1e-12:
            raise ValueError('Static fixture requires zero velocity')
        command_low, command_high = interface.admitted_target_bounds(state.joint_position, dt)
        requested = np.clip(requested, command_low, command_high)
        snapshot = self.model.snapshot(state.q, state.v)
        com_jac, com_bias = self.com_kinematics(state)
        self._static_fixture = bool(static_equilibrium)
        try:
            reference = self.reference(state, snapshot, requested)
        finally:
            self._static_fixture = False
        feet = np.flatnonzero(state.contacts)
        nf, nv = 3 * len(feet), 24
        nz = nv + nf
        jac = snapshot['foot_jacobian'][feet].reshape(nf, nv)
        jdot_v = (snapshot['foot_jacobian_dot'][feet] @ state.v).reshape(nf)
        contact_acc = -jac @ state.v / cfg.contact_velocity_decay_s
        torque_map = np.c_[snapshot['mass_matrix'][self.model.v_indices], -jac[:, self.model.v_indices].T]
        torque_bias = snapshot['nonlinear'][self.model.v_indices]
        selector = np.zeros((18, nz))
        selector[np.arange(18), self.model.v_indices] = 1.
        command_map, command_bias = implicit_command_map(torque_map, torque_bias,
            state.joint_position, state.joint_velocity, interface.kp, interface.kd, selector, dt)
        hessian = np.diag(np.r_[np.full(nv, cfg.acceleration_regularization),
                                  np.full(nf, cfg.force_regularization)])
        linear = np.zeros(nz)

        def task(matrix, target, weight):
            nonlocal hessian, linear
            hessian += weight * matrix.T @ matrix
            linear -= weight * matrix.T @ target

        task(torque_map, -torque_bias, cfg.torque_regularization)
        task(np.c_[com_jac, np.zeros((3, nf))], reference['com_acceleration'] - com_bias, cfg.com_weight)
        angular, angular_target = self.angular_task(state, snapshot, reference)
        if (angular.shape != (3, nv) or np.asarray(angular_target).shape != (3,) or
                not np.isfinite(angular).all() or not np.isfinite(angular_target).all()):
            raise ValueError('Finite angular task with full generalized acceleration required')
        task(np.c_[angular, np.zeros((3, nf))], angular_target, cfg.orientation_weight)
        task(selector, reference['joint_acceleration'], cfg.posture_weight)
        foot_ids = [self.model.names.index(n) for n in self.model.feet]
        swing = reference.get('swing_foot_target_world')
        if cfg.swing_foot_weight > 1e-7 and swing is not None and len(feet):
            for foot in np.flatnonzero(~state.contacts):
                jf = snapshot['foot_jacobian'][foot]
                desired = (cfg.swing_foot_kp_s2 * (swing[foot] - snapshot['body_positions_world'][foot_ids[foot]])
                           - cfg.swing_foot_kd_s * (jf @ state.v))
                task(np.c_[jf, np.zeros((3, nf))],
                     desired - snapshot['foot_jacobian_dot'][foot] @ state.v, cfg.swing_foot_weight)
        clearance_points = {}
        if cfg.nonfoot_clearance_weight > 1e-7 and len(feet):
            frames = frame_point_kinematics(self.model, state.q, state.v,
                ('FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'))
            ground = float(np.median(snapshot['body_positions_world'][np.asarray(foot_ids)[feet], 2]))
            for name, frame in frames.items():
                clearance_points[name] = frame['position'].tolist()
                target_z = ground + cfg.nonfoot_clearance_m
                z, vz = frame['position'][2], frame['velocity'][2]
                if z + dt * vz <= target_z + .12 or vz < -.5:
                    desired = np.clip(cfg.nonfoot_clearance_kp_s2 * (target_z - z)
                                      - cfg.nonfoot_clearance_kd_s * vz, -80., 80.)
                    block = np.zeros((1, nz)); block[0, :nv] = frame['jacobian'][2]
                    task(block, np.array([desired - frame['jacobian_dot'][2] @ state.v]),
                         cfg.nonfoot_clearance_weight)
        matrices, lows, highs = [], [], []

        def constraint(matrix, low, high=None):
            matrices.append(np.atleast_2d(matrix))
            lows.append(np.atleast_1d(low))
            highs.append(np.atleast_1d(low if high is None else high))

        constraint(np.c_[snapshot['mass_matrix'][:6], -jac[:, :6].T], -snapshot['nonlinear'][:6])
        if nf:
            constraint(np.c_[jac, np.zeros((nf, nf))], contact_acc - jdot_v)
        for i in range(len(feet)):
            angles = np.arange(cfg.friction_facets) * 2 * np.pi / cfg.friction_facets
            block = np.zeros((cfg.friction_facets, nz))
            block[:, nv + 3*i] = np.cos(angles); block[:, nv + 3*i + 1] = np.sin(angles)
            block[:, nv + 3*i + 2] = -interface.friction * np.cos(np.pi / cfg.friction_facets)
            constraint(block, np.full(cfg.friction_facets, -np.inf), np.zeros(cfg.friction_facets))
            block = np.zeros(nz); block[nv + 3*i + 2] = 1.
            constraint(block, 0., np.inf)
        constraint(torque_map, -interface.effort - torque_bias, interface.effort - torque_bias)
        constraint(command_map, command_low - command_bias, command_high - command_bias)
        if self.preserve_arm_command:
            constraint(command_map[12:], requested[12:] - command_bias[12:])
        qbase = state.joint_position + dt * state.joint_velocity
        alow = np.maximum((-interface.velocity-state.joint_velocity)/dt,
                          (interface.limits[:, 0]-qbase)/dt**2)
        ahigh = np.minimum((interface.velocity-state.joint_velocity)/dt,
                           (interface.limits[:, 1]-qbase)/dt**2)
        constraint(selector, alow, ahigh)
        if static_equilibrium:
            constraint(np.c_[np.eye(nv), np.zeros((nv, nf))], np.zeros(nv))
        A, low, high = np.vstack(matrices), np.concatenate(lows), np.concatenate(highs)
        plan = dict(accepted=False, contacts=state.contacts.copy(), reference=reference,
            formulation='backward-Euler-PD-command-QP-unsaturated-branch', native_execution=False,
            nonfoot_clearance_points_world=clearance_points,
            previous_command_target=interface.previous_command.tolist(),
            admitted_reference=requested.tolist(), decision_variables=nz)
        if np.any(low > high):
            plan.update(solver_status='primal infeasible: conflicting original joint bounds',
                        rejection='Original joint position/velocity constraints conflict')
            return plan
        scale = np.r_[np.full(nv, 10.), np.full(nf, self.model.mass*9.81/4.)]
        scaled = A * scale[None, :]
        row_scale = 1. / np.maximum(np.max(abs(scaled), axis=1), 1e-12)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian*scale[:, None]*scale[None, :])), q=linear*scale,
            A=sparse.csc_matrix(scaled*row_scale[:, None]), l=low*row_scale, u=high*row_scale,
            eps_abs=cfg.solver_eps, eps_rel=cfg.solver_eps, max_iter=cfg.solver_max_iterations,
            polish=True, verbose=False)
        result = solver.solve()
        plan.update(solver_status=result.info.status, solver_iterations=result.info.iter,
                    solver_runtime_s=result.info.run_time, wall_runtime_s=time.perf_counter()-started)
        if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
            plan['rejection'] = 'Solver did not return a finite solved result'
            return plan
        x = result.x * scale; acceleration = x[:nv]
        torque = torque_map @ x + torque_bias
        command = command_map @ x + command_bias
        vnext = state.joint_velocity + dt * acceleration[self.model.v_indices]
        qnext = state.joint_position + dt * vnext
        implicit_torque = interface.kp*(command-qnext) - interface.kd*vnext
        force = np.zeros((4, 3)); force[feet] = x[nv:].reshape(-1, 3)
        violation = lambda a: float(max(0., np.max(a, initial=0.)))
        residuals = dict(
            dynamics=float(np.max(abs(snapshot['mass_matrix']@acceleration+snapshot['nonlinear']
                - self.selection@torque - jac.T@x[nv:]))),
            active_contact_acceleration=float(np.max(abs(jac@acceleration+jdot_v-contact_acc), initial=0.)),
            implicit_pd_torque=float(np.max(abs(implicit_torque-torque))),
            normal_force=violation(-force[:, 2]),
            circular_friction_cone=violation(np.linalg.norm(force[:, :2], axis=1)-interface.friction*force[:, 2]),
            torque_bounds=violation(abs(torque)-interface.effort),
            command_bounds=max(violation(command_low-command), violation(command-command_high)),
            predicted_position_bounds=max(violation(interface.limits[:, 0]-qnext), violation(qnext-interface.limits[:, 1])),
            predicted_velocity_bounds=violation(abs(vnext)-interface.velocity),
            all_linear_constraints=max(violation(low-A@x), violation(A@x-high)))
        if self.preserve_arm_command:
            residuals['prescribed_arm_target'] = float(np.max(abs(command[12:]-requested[12:])))
        plan.update(acceleration=acceleration, torque=torque, force_world=force,
            joint_position_target=command, normalized_action=interface.normalized_action(command),
            predicted_joint_position=qnext, predicted_joint_velocity=vnext,
            predicted_com_acceleration=com_jac@acceleration+com_bias, residuals=residuals)
        plan['accepted'] = max(residuals.values()) <= cfg.residual_tolerance
        if not plan['accepted']:
            plan['rejection'] = 'Independent implicit-drive/physical constraint residual failed'
        return plan
