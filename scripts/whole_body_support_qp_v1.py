"""Contact-aware full inverse dynamics through the existing position interface.

CPU prototype. Solving this QP is NOT evidence of a physically successful
controller: implicit-PD torque realization and contact response need native
execution. It uses all loaded rigid bodies and never adds external base forces.
Airborne feet have exactly zero force. Joint and force commands are original
hardware/interface limited; posture, COM and orientation tracking are costs.
"""
from dataclasses import dataclass, asdict
import time

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse

from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


@dataclass(frozen=True)
class SupportSettings:
    control_dt_s: float = .01
    contact_velocity_decay_s: float = .06
    com_kp_s2: float = 50.
    com_kd_s: float = 14.
    orientation_kp_s2: float = 60.
    orientation_kd_s: float = 15.
    posture_kp_s2: float = 40.
    posture_kd_s: float = 12.
    com_weight: float = 20.
    orientation_weight: float = 10.
    posture_weight: float = .20
    acceleration_regularization: float = 1e-5
    torque_regularization: float = 1e-4
    force_regularization: float = 1e-5
    force_distribution_weight: float = 0.
    force_wrench_weight: float = 0.
    friction_facets: int = 8
    solver_eps: float = 1e-7
    solver_max_iterations: int = 10000
    residual_tolerance: float = 2e-4
    # Optional soft touchdown task.  The tiny default keeps the original
    # standing/support QP numerically equivalent unless a landing controller
    # explicitly opts in with a larger weight.
    swing_foot_weight: float = 1e-8
    swing_foot_kp_s2: float = 80.
    swing_foot_kd_s: float = 18.
    swing_foot_clearance_m: float = .025
    # Optional capture-point/VHSIP horizontal offset for inactive feet.  Zero
    # preserves the original posture-relative swing target.
    swing_foot_capture_gain: float = 0.
    swing_foot_capture_max_m: float = .25
    # Optional soft clearance task for lower-leg body links.  It is disabled
    # by default; landing trials can enable it to avoid a thigh/body link
    # entering the support plane after the first foot contact.
    nonfoot_clearance_weight: float = 0.
    nonfoot_clearance_hard: bool = False
    nonfoot_clearance_m: float = .08
    nonfoot_clearance_kp_s2: float = 120.
    nonfoot_clearance_kd_s: float = 25.


class PositionPDInterface:
    """Instantaneous reachable torque set, not a full-period torque guarantee."""
    def __init__(self, limits, stiffness, damping, effort, velocity, friction):
        self.limits = np.array(limits, dtype=np.float64)
        self.kp, self.kd, self.effort, self.velocity = [np.asarray(x, dtype=np.float64)
            for x in (stiffness, damping, effort, velocity)]
        self.friction = float(friction)
        if (self.limits.shape != (18, 2) or np.any(self.limits[:, 0] >= self.limits[:, 1]) or
                any(x.shape != (18,) or not np.isfinite(x).all()
                    for x in [self.kp, self.kd, self.effort, self.velocity]) or
                np.any(self.kp <= 0) or np.any(self.kd < 0) or
                np.any(self.effort <= 0) or np.any(self.velocity <= 0) or
                not np.isfinite(self.friction) or self.friction <= 0):
            raise ValueError('Invalid original joint/interface limits')

    def admitted_target_bounds(self, q, control_dt_s):
        """Legacy measured-position-centered controller envelope (not hardware slew)."""
        step = self.velocity * float(control_dt_s)
        return (np.maximum(self.limits[:, 0], q-step),
                np.minimum(self.limits[:, 1], q+step))

    def torque_bounds(self, q, v, control_dt_s=None):
        """Return torque bounds consistent with the native position interface.

        With ``control_dt_s`` supplied, intersect the joint-limit torque set
        with the set reachable by a position target admitted for one control
        tick. This keeps the QP request, target admission, and native action
        on the same interface instead of solving an aggressive torque and
        clipping it only after the solve.
        """
        endpoints = self.kp[:, None] * (self.limits - q[:, None]) - self.kd[:, None] * v[:, None]
        low, high = np.clip(endpoints, -self.effort[:, None], self.effort[:, None]).T
        if control_dt_s is not None:
            dt = float(control_dt_s)
            if not np.isfinite(dt) or dt <= 0.:
                raise ValueError('control_dt_s must be positive and finite')
            admitted_low = np.maximum(self.limits[:, 0], q - self.velocity * dt)
            admitted_high = np.minimum(self.limits[:, 1], q + self.velocity * dt)
            admitted_endpoints = self.kp[:, None] * (
                np.stack((admitted_low, admitted_high), axis=1) - q[:, None]
            ) - self.kd[:, None] * v[:, None]
            low = np.maximum(low, admitted_endpoints[:, 0])
            high = np.minimum(high, admitted_endpoints[:, 1])
        return low, high

    def target_from_torque(self, torque, q, v):
        target = np.clip(q + (torque + self.kd*v)/self.kp,
                         self.limits[:, 0], self.limits[:, 1])
        achieved_instantaneously = np.clip(self.kp*(target-q) - self.kd*v,
                                           -self.effort, self.effort)
        return target, achieved_instantaneously

    def normalized_action(self, target):
        return 2.*(target-self.limits[:, 0]) / np.diff(self.limits, axis=1)[:, 0] - 1.


class WholeBodySupportQP:
    def __init__(self, interface, settings=SupportSettings()):
        declared = asdict(settings)
        for name, value in declared.items():
            if isinstance(value, bool):
                continue
            if (not np.isfinite(value) or value < 0 or
                    (name not in ('swing_foot_weight', 'nonfoot_clearance_weight',
                                  'force_distribution_weight',
                                  'force_wrench_weight',
                                  'swing_foot_capture_gain') and value == 0)):
                raise ValueError('Finite nonnegative control settings required; all non-swing terms must be positive')
        if settings.swing_foot_capture_max_m <= 0.:
            raise ValueError('swing_foot_capture_max_m must be positive')
        if settings.friction_facets < 4 or settings.friction_facets % 2:
            raise ValueError('Even friction polygon with at least four facets required')
        self.model = LoadedPinocchioModelV1()
        self.interface, self.settings = interface, settings
        self.nv, self.nt, self.nf = 24, 18, 12
        self.nz = self.nv + self.nt + self.nf
        self.a_slice, self.t_slice, self.f_slice = slice(0,24), slice(24,42), slice(42,54)
        self.selection = np.zeros((24, 18))
        self.selection[self.model.v_indices, np.arange(18)] = 1.

    def com_kinematics(self, state):
        m, d = self.model.model, self.model.data
        jac = pin.jacobianCenterOfMass(m, d, state.q).copy()
        pin.centerOfMass(m, d, state.q, state.v, np.zeros(m.nv))
        return jac, d.acom[0].copy()

    def reference(self, state, snapshot, joint_reference):
        """Support height follows the commanded loaded posture, not a fixed height.

        A horizontal COM target lies above the measured support centroid. This
        one-step reference still needs compliant touchdown timing/anchoring in
        the native controller; it is not a complete airborne landing planner.
        """
        r = state.root_rotation_world
        yaw = np.arctan2(r[1, 0], r[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        upright = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        desired_com = snapshot['com_world'].copy()
        if state.contacts.any():
            q_reference = state.q.copy()
            q_reference[:3] = 0.
            q_reference[3:7] = pin.Quaternion(upright).coeffs()
            q_reference[self.model.q_indices] = joint_reference
            fk_positions, _ = self.model.poses(q_reference)
            feet_ids = [self.model.names.index(n) for n in self.model.feet]
            mean_reference_feet = np.array(fk_positions)[feet_ids][state.contacts].mean(0)
            reference_com = pin.centerOfMass(self.model.model, self.model.data, q_reference).copy()
            measured_feet = snapshot['body_positions_world'][feet_ids]
            center = measured_feet[state.contacts].mean(0)
            desired_com[:2] = center[:2]
            desired_com[2] = center[2] + reference_com[2] - mean_reference_feet[2]
        cfg = self.settings
        desired_com_accel = (cfg.com_kp_s2*(desired_com - snapshot['com_world']) -
                             cfg.com_kd_s*snapshot['com_velocity_world'])
        desired_angular_accel_body = (cfg.orientation_kp_s2*pin.log3(r.T @ upright) -
                                     cfg.orientation_kd_s*state.v[3:6])
        desired_joint_accel = (cfg.posture_kp_s2*(joint_reference-state.joint_position) -
                              cfg.posture_kd_s*state.joint_velocity)
        # A landing controller may opt into a soft target for feet that are
        # not currently in contact.  The target is expressed in the loaded
        # robot/world model and is only a cost; it never creates a ground
        # force or changes the observed contact mask.
        swing_targets = np.zeros((4, 3))
        if state.contacts.any():
            feet_ids = [self.model.names.index(n) for n in self.model.feet]
            reference_feet = np.array(fk_positions)[feet_ids]
            measured_feet = snapshot['body_positions_world'][feet_ids]
            active_reference = reference_feet[state.contacts]
            active_measured = measured_feet[state.contacts]
            reference_center = active_reference.mean(0)
            measured_center = active_measured.mean(0)
            swing_targets = measured_center + reference_feet - reference_center
            swing_targets[:, 2] = measured_center[2] + cfg.swing_foot_clearance_m
            swing_targets[state.contacts] = measured_feet[state.contacts]
            if cfg.swing_foot_capture_gain > 0. and (~state.contacts).any():
                # A capture-point/VHSIP-style offset moves only feet that are
                # not currently supporting the body.  The support QP still
                # treats this as a soft swing task; it does not invent a
                # reaction force or claim that touchdown has occurred.
                height = max(float(snapshot['com_world'][2] - measured_center[2]), .10)
                capture = (cfg.swing_foot_capture_gain *
                           np.sqrt(height / 9.81) * snapshot['com_velocity_world'][:2])
                norm = float(np.linalg.norm(capture))
                if norm > cfg.swing_foot_capture_max_m:
                    capture *= cfg.swing_foot_capture_max_m / norm
                swing_targets[~state.contacts, :2] += capture
        return dict(com_position=desired_com, com_acceleration=desired_com_accel,
                    angular_acceleration_body=desired_angular_accel_body,
                    joint_acceleration=desired_joint_accel,
                    swing_foot_target_world=swing_targets)

    def solve(self, state, joint_reference, *, static_equilibrium=False):
        """Return a separately verified model-feasible plan, or explicit failure.

        static_equilibrium constrains ALL accelerations to zero and requires
        zero velocity. It tests support existence in the model, not a native
        standing trial. Normal solves retain measured state and contact mask.
        """
        started = time.perf_counter()
        cfg, interface = self.settings, self.interface
        joint_reference = np.asarray(joint_reference, dtype=np.float64)
        if joint_reference.shape != (18,) or not np.isfinite(joint_reference).all():
            raise ValueError('18 finite posture targets required')
        if np.any(joint_reference < interface.limits[:,0]-1e-6) or np.any(joint_reference > interface.limits[:,1]+1e-6):
            raise ValueError('Posture target outside original position bounds')
        if static_equilibrium and np.max(np.abs(state.v)) > 1e-12:
            raise ValueError('Static check requires explicitly zeroed model velocities')
        snapshot = self.model.snapshot(state.q, state.v)
        com_jac, com_bias = self.com_kinematics(state)
        reference = self.reference(state, snapshot, joint_reference)
        jac = snapshot['foot_jacobian'].reshape(12,24)
        jdot_v = (snapshot['foot_jacobian_dot'] @ state.v).reshape(12)
        foot_velocity = jac @ state.v
        contact_acceleration = -foot_velocity / cfg.contact_velocity_decay_s
        # Positive definite quadratic costs; body/COM/posture tracking is soft.
        diagonal = np.r_[np.full(24,cfg.acceleration_regularization),
                         np.full(18,cfg.torque_regularization), np.full(12,cfg.force_regularization)]
        hessian, linear = np.diag(diagonal), np.zeros(self.nz)

        def task(matrix, target, weight):
            nonlocal hessian, linear
            hessian += weight * matrix.T @ matrix
            linear -= weight * matrix.T @ target

        task_matrix = np.zeros((3,self.nz)); task_matrix[:,:24] = com_jac
        task(task_matrix, reference['com_acceleration'] - com_bias, cfg.com_weight)
        task_matrix = np.zeros((3,self.nz)); task_matrix[:,3:6] = np.eye(3)
        task(task_matrix, reference['angular_acceleration_body'], cfg.orientation_weight)
        task_matrix = np.zeros((18,self.nz)); task_matrix[np.arange(18),self.model.v_indices] = 1.
        task(task_matrix, reference['joint_acceleration'], cfg.posture_weight)
        matrices, lowers, uppers = [], [], []

        def constraint(matrix, lower, upper=None):
            matrices.append(np.atleast_2d(matrix))
            lowers.append(np.atleast_1d(lower))
            uppers.append(np.atleast_1d(lower if upper is None else upper))

        dynamics = np.c_[snapshot['mass_matrix'], -self.selection, -jac.T]
        constraint(dynamics, -snapshot['nonlinear'])
        for i, contacting in enumerate(state.contacts):
            rows = slice(3*i,3*i+3)
            if contacting:
                block = np.zeros((3,self.nz)); block[:,:24] = jac[rows]
                constraint(block, contact_acceleration[rows]-jdot_v[rows])
                # Inscribed polygon: every accepted force satisfies the circular
                # Coulomb cone; do not use an outer box allowing sqrt(2)*mu.
                theta = np.arange(cfg.friction_facets)*2*np.pi/cfg.friction_facets
                block = np.zeros((cfg.friction_facets,self.nz))
                block[:,42+3*i] = np.cos(theta)
                block[:,43+3*i] = np.sin(theta)
                block[:,44+3*i] = -interface.friction*np.cos(np.pi/cfg.friction_facets)
                constraint(block, np.full(cfg.friction_facets,-np.inf), np.zeros(cfg.friction_facets))
                block = np.zeros(self.nz); block[44+3*i] = 1.
                constraint(block, 0., np.inf)
            else:
                block = np.zeros((3,self.nz)); block[:,42+3*i:45+3*i] = np.eye(3)
                constraint(block, np.zeros(3))
        # Keep the support QP inside the same one-tick absolute-position
        # command set used by the native evaluator.
        torque_low, torque_high = interface.torque_bounds(
            state.joint_position, state.joint_velocity, cfg.control_dt_s)
        block = np.zeros((18,self.nz)); block[:,self.t_slice] = np.eye(18)
        constraint(block, torque_low, torque_high)
        # Original hardware velocity limits applied to a one-control-step prediction.
        block = np.zeros((18,self.nz)); block[np.arange(18),self.model.v_indices] = cfg.control_dt_s
        constraint(block, -interface.velocity-state.joint_velocity,
                   interface.velocity-state.joint_velocity)
        block = np.zeros((18,self.nz)); block[np.arange(18),self.model.v_indices] = .5*cfg.control_dt_s**2
        next_without_acceleration = state.joint_position + cfg.control_dt_s*state.joint_velocity
        constraint(block, interface.limits[:,0]-next_without_acceleration,
                   interface.limits[:,1]-next_without_acceleration)
        if static_equilibrium:
            block = np.zeros((24,self.nz)); block[:,:24] = np.eye(24)
            constraint(block,np.zeros(24))
        matrix = np.vstack(matrices); low, high = np.concatenate(lowers), np.concatenate(uppers)
        # Equivalent numerical coordinates, preserving every physical bound and
        # objective weight. Unscaled next-position rows carry dt^2/2 ~=5e-5,
        # whereas dynamics rows carry mass/inertia coefficients. Explicit row
        # and variable scaling avoids treating these as differently accurate
        # constraints. Verify residuals again in ORIGINAL physical coordinates.
        scale = np.r_[np.full(24,10.),interface.effort,
                      np.full(12,self.model.mass*9.81/4.)]
        scaled_matrix = matrix*scale[None,:]
        row_scale = 1./np.maximum(np.max(np.abs(scaled_matrix),axis=1),1e-12)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian*scale[:,None]*scale[None,:])),q=linear*scale,
                     A=sparse.csc_matrix(scaled_matrix*row_scale[:,None]),l=low*row_scale,u=high*row_scale,
                     eps_abs=cfg.solver_eps,eps_rel=cfg.solver_eps,
                     max_iter=cfg.solver_max_iterations,polish=True,verbose=False)
        solved = solver.solve()
        plan = dict(accepted=False,solver_status=solved.info.status,
                    solver_iterations=solved.info.iter,solver_runtime_s=solved.info.run_time,
                    wall_runtime_s=time.perf_counter()-started,
                    contacts=state.contacts.copy(),reference=reference,
                    static_model_check=static_equilibrium,native_execution=False)
        if solved.info.status_val != 1 or solved.x is None or not np.isfinite(solved.x).all():
            plan['rejection']='Solver did not return a finite solved result'
            return plan
        x = solved.x*scale
        violation = max(float(np.max(low-matrix@x)),float(np.max(matrix@x-high)),0.)
        a, torque, force = x[self.a_slice],x[self.t_slice],x[self.f_slice].reshape(4,3)
        target, instantaneous_torque = interface.target_from_torque(torque,state.joint_position,state.joint_velocity)
        residuals = dict(all_linear_constraints=violation,
            dynamics=float(np.max(np.abs(dynamics@x+snapshot['nonlinear']))),
            active_contact_acceleration=float(np.max(np.abs((jac@a+jdot_v-contact_acceleration).reshape(4,3)[state.contacts]))) if state.contacts.any() else 0.,
            inactive_foot_force=float(np.max(np.abs(force[~state.contacts]))) if (~state.contacts).any() else 0.,
            circular_friction_cone=float(max(0.,np.max(np.linalg.norm(force[:,:2],axis=1)-interface.friction*force[:,2]))),
            instantaneous_pd_torque=float(np.max(np.abs(instantaneous_torque-torque))),
            admitted_target_bounds=float(max(
                0.,
                np.max(interface.admitted_target_bounds(state.joint_position,cfg.control_dt_s)[0]-target),
                np.max(target-interface.admitted_target_bounds(state.joint_position,cfg.control_dt_s)[1]))))
        plan.update(residuals=residuals,acceleration=a,torque=torque,force_world=force,
                    joint_position_target=target,normalized_action=interface.normalized_action(target),
                    predicted_com_acceleration=com_jac@a+com_bias,
                    predicted_joint_velocity=state.joint_velocity+cfg.control_dt_s*a[self.model.v_indices],
                    predicted_joint_position=next_without_acceleration+.5*cfg.control_dt_s**2*a[self.model.v_indices])
        plan['accepted'] = max(residuals.values()) <= cfg.residual_tolerance
        if not plan['accepted']:
            plan['rejection']='Independent residual verification failed'
        return plan
