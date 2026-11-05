"""Equivalent inverse dynamics QP with torques/inactive forces eliminated.

This is the same physical model, tracking cost and limits as the full-coordinate
prototype. Torque is recovered by the actuated equations of motion; the six
unactuated equations remain hard constraints. Inactive forces are identically
zero. Reducing redundant variables improves numerical conditioning without
relaxing constraints or adding external forces.
"""
import time

import numpy as np
import osqp
from scipy import sparse

from scripts.whole_body_support_qp_v1 import WholeBodySupportQP


class ReducedWholeBodySupportQP(WholeBodySupportQP):
    def solve(self, state, joint_reference, *, static_equilibrium=False):
        started=time.perf_counter()
        cfg, interface=self.settings,self.interface
        joint_reference=np.asarray(joint_reference,dtype=np.float64)
        if joint_reference.shape!=(18,) or not np.isfinite(joint_reference).all():
            raise ValueError('18 finite posture targets required')
        if np.any(joint_reference<interface.limits[:,0]-1e-6) or np.any(joint_reference>interface.limits[:,1]+1e-6):
            raise ValueError('Posture target outside original bounds')
        if static_equilibrium and np.max(abs(state.v))>1e-12:
            raise ValueError('Static fixture requires explicit zero velocity')
        snapshot=self.model.snapshot(state.q,state.v)
        com_jac,com_bias=self.com_kinematics(state)
        reference=self.reference(state,snapshot,joint_reference)
        feet=np.flatnonzero(state.contacts);nforce=3*len(feet);nz=24+nforce
        feet_ids = [self.model.names.index(n) for n in self.model.feet]
        jac=snapshot['foot_jacobian'][feet].reshape(nforce,24)
        jdot_v=(snapshot['foot_jacobian_dot'][feet]@state.v).reshape(nforce)
        contact_acceleration=-(jac@state.v)/cfg.contact_velocity_decay_s
        torque_map=np.c_[snapshot['mass_matrix'][self.model.v_indices],-jac[:,self.model.v_indices].T]
        torque_bias=snapshot['nonlinear'][self.model.v_indices]
        hessian=np.diag(np.r_[np.full(24,cfg.acceleration_regularization),np.full(nforce,cfg.force_regularization)])
        linear=np.zeros(nz)

        def task(matrix,target,weight):
            nonlocal hessian,linear
            hessian+=weight*matrix.T@matrix;linear-=weight*matrix.T@target

        task(torque_map,-torque_bias,cfg.torque_regularization)
        block=np.zeros((3,nz));block[:,:24]=com_jac
        task(block,reference['com_acceleration']-com_bias,cfg.com_weight)
        block=np.zeros((3,nz));block[:,3:6]=np.eye(3)
        task(block,reference['angular_acceleration_body'],cfg.orientation_weight)
        block=np.zeros((18,nz));block[np.arange(18),self.model.v_indices]=1.
        task(block,reference['joint_acceleration'],cfg.posture_weight)
        # Optional touchdown shaping for feet that are not yet supporting the
        # body.  This remains a soft acceleration cost: only observed contacts
        # receive reaction-force variables, so the QP cannot invent a contact
        # while it prepares the swing leg for a later landing.
        swing_weight = float(cfg.swing_foot_weight)
        swing_targets = reference.get('swing_foot_target_world')
        if swing_weight > 1e-7 and swing_targets is not None and len(feet):
            for foot_index in range(4):
                if state.contacts[foot_index]:
                    continue
                jac_foot = snapshot['foot_jacobian'][foot_index]
                jdot_v_foot = snapshot['foot_jacobian_dot'][foot_index] @ state.v
                foot_position = snapshot['body_positions_world'][foot_ids[foot_index]]
                foot_velocity = jac_foot @ state.v
                desired_foot_acceleration = (
                    cfg.swing_foot_kp_s2 *
                    (swing_targets[foot_index] - foot_position)
                    - cfg.swing_foot_kd_s * foot_velocity)
                swing_block = np.zeros((3, nz)); swing_block[:, :24] = jac_foot
                task(swing_block, desired_foot_acceleration - jdot_v_foot,
                     swing_weight)
        # Keep the lower-leg body links above the measured support plane.  The
        # task is deliberately soft and only activates near the plane or while
        # a thigh is moving downward; it does not pretend to model a new body
        # contact.  This targets the audited failure mode where a foot lands
        # first and a thigh touches the ground later.
        clearance_weight = float(cfg.nonfoot_clearance_weight)
        if clearance_weight > 1e-7 and len(feet):
            active_positions = snapshot['body_positions_world'][feet_ids][state.contacts]
            ground_z = float(np.median(active_positions[:, 2])) if len(active_positions) else None
            if ground_z is not None:
                for link_name in ('FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'):
                    frame_id = self.model.body_frames[link_name]
                    link_jac = pin.getFrameJacobian(self.model.model, self.model.data,
                        frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3].copy()
                    link_jdot = pin.getFrameJacobianTimeVariation(
                        self.model.model, self.model.data, frame_id,
                        pin.ReferenceFrame.WORLD)[:3].copy()
                    link_position = self.model.data.oMf[frame_id].translation.copy()
                    link_velocity = link_jac @ state.v
                    target_z = ground_z + float(cfg.nonfoot_clearance_m)
                    predicted_z = link_position[2] + cfg.control_dt_s * link_velocity[2]
                    if (predicted_z <= target_z + .12 or link_velocity[2] < -.5):
                        desired_z_acceleration = (
                            cfg.nonfoot_clearance_kp_s2 * (target_z - link_position[2])
                            - cfg.nonfoot_clearance_kd_s * link_velocity[2])
                        desired_z_acceleration = float(np.clip(desired_z_acceleration, -80., 80.))
                        clearance_block = np.zeros((1, nz)); clearance_block[0, :24] = link_jac[2]
                        jdot_v_link = float(link_jdot[2] @ state.v)
                        task(clearance_block, desired_z_acceleration - jdot_v_link,
                             clearance_weight)
        matrices,lowers,uppers=[],[],[]

        def constraint(matrix,low,high=None):
            matrices.append(np.atleast_2d(matrix));lowers.append(np.atleast_1d(low))
            uppers.append(np.atleast_1d(low if high is None else high))

        constraint(np.c_[snapshot['mass_matrix'][:6],-jac[:,:6].T],-snapshot['nonlinear'][:6])
        if nforce:
            constraint(np.c_[jac,np.zeros((nforce,nforce))],contact_acceleration-jdot_v)
        for i in range(len(feet)):
            theta=np.arange(cfg.friction_facets)*2*np.pi/cfg.friction_facets
            block=np.zeros((cfg.friction_facets,nz))
            block[:,24+3*i]=np.cos(theta);block[:,25+3*i]=np.sin(theta)
            block[:,26+3*i]=-interface.friction*np.cos(np.pi/cfg.friction_facets)
            constraint(block,np.full(cfg.friction_facets,-np.inf),np.zeros(cfg.friction_facets))
            block=np.zeros(nz);block[26+3*i]=1.
            constraint(block,0.,np.inf)
        torque_low,torque_high=interface.torque_bounds(state.joint_position,state.joint_velocity)
        constraint(torque_map,torque_low-torque_bias,torque_high-torque_bias)
        next_position=state.joint_position+cfg.control_dt_s*state.joint_velocity
        acceleration_low=np.maximum((-interface.velocity-state.joint_velocity)/cfg.control_dt_s,
                                    (interface.limits[:,0]-next_position)/(.5*cfg.control_dt_s**2))
        acceleration_high=np.minimum((interface.velocity-state.joint_velocity)/cfg.control_dt_s,
                                     (interface.limits[:,1]-next_position)/(.5*cfg.control_dt_s**2))
        block=np.zeros((18,nz));block[np.arange(18),self.model.v_indices]=1.
        constraint(block,acceleration_low,acceleration_high)
        if static_equilibrium:constraint(np.c_[np.eye(24),np.zeros((24,nforce))],np.zeros(24))
        matrix=np.vstack(matrices);low,high=np.concatenate(lowers),np.concatenate(uppers)
        plan=dict(accepted=False,contacts=state.contacts.copy(),reference=reference,
                  static_model_check=static_equilibrium,native_execution=False,
                  formulation='actuated_torques_and_inactive_forces_eliminated',decision_variables=nz)
        if np.any(low>high):
            plan.update(solver_status='primal infeasible: conflicting original position/velocity bounds',
                        rejection='Predicted original joint position and velocity bounds conflict',
                        solver_iterations=0,solver_runtime_s=0.,wall_runtime_s=time.perf_counter()-started)
            return plan
        scale=np.r_[np.full(24,10.),np.full(nforce,self.model.mass*9.81/4.)]
        scaled=matrix*scale[None,:];row_scale=1./np.maximum(np.max(abs(scaled),axis=1),1e-12)
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian*scale[:,None]*scale[None,:])),q=linear*scale,
                     A=sparse.csc_matrix(scaled*row_scale[:,None]),l=low*row_scale,u=high*row_scale,
                     eps_abs=cfg.solver_eps,eps_rel=cfg.solver_eps,
                     max_iter=cfg.solver_max_iterations,polish=True,verbose=False)
        solved=solver.solve()
        plan.update(solver_status=solved.info.status,solver_iterations=solved.info.iter,
                    solver_runtime_s=solved.info.run_time,wall_runtime_s=time.perf_counter()-started)
        if solved.info.status_val!=1 or solved.x is None or not np.isfinite(solved.x).all():
            plan['rejection']='Solver did not return a finite solved result';return plan
        x=solved.x*scale;a=x[:24];force=np.zeros((4,3));force[feet]=x[24:].reshape(-1,3)
        torque=torque_map@x+torque_bias
        target,pd_torque=interface.target_from_torque(torque,state.joint_position,state.joint_velocity)
        whole_dynamics=(snapshot['mass_matrix']@a+snapshot['nonlinear']-self.selection@torque-
                        snapshot['foot_jacobian'].reshape(12,24).T@force.ravel())
        # Verify physical quantities, including original position/velocity units;
        # acceleration constraints were scaled only to improve the solve.
        qnext=next_position+.5*cfg.control_dt_s**2*a[self.model.v_indices]
        vnext=state.joint_velocity+cfg.control_dt_s*a[self.model.v_indices]
        residuals=dict(dynamics=float(np.max(abs(whole_dynamics))),
            active_contact_acceleration=float(np.max(abs(jac@a+jdot_v-contact_acceleration))) if nforce else 0.,
            inactive_foot_force=float(np.max(abs(force[~state.contacts]))) if (~state.contacts).any() else 0.,
            circular_friction_cone=float(max(0.,np.max(np.linalg.norm(force[:,:2],axis=1)-interface.friction*force[:,2]))),
            instantaneous_pd_torque=float(np.max(abs(pd_torque-torque))),
            torque_bounds=float(max(0.,np.max(torque_low-torque),np.max(torque-torque_high))),
            predicted_position_bounds=float(max(0.,np.max(interface.limits[:,0]-qnext),np.max(qnext-interface.limits[:,1]))),
            predicted_velocity_bounds=float(max(0.,np.max(abs(vnext)-interface.velocity))))
        # The original linear system additionally enforces the inscribed friction
        # polygon. Its row violations are returned separately from unitful checks.
        violation=max(0.,float(np.max(low-matrix@x)),float(np.max(matrix@x-high)))
        residuals['all_linear_constraints']=violation
        plan.update(residuals=residuals,acceleration=a,torque=torque,force_world=force,
                    joint_position_target=target,normalized_action=interface.normalized_action(target),
                    predicted_com_acceleration=com_jac@a+com_bias,predicted_joint_velocity=vnext,
                    predicted_joint_position=qnext)
        plan['accepted']=max(residuals.values())<=cfg.residual_tolerance
        if not plan['accepted']:plan['rejection']='Independent residual verification failed'
        return plan
