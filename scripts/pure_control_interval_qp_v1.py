"""Pure-control full18 execution QP with authored nonfoot surfaces.

One-step predictive WBC, NOT a sequence-optimizing MPC or validated teacher.
Optimize a held native position command with constraints inside the interval.

Sixteen implicit drive steps are condensed into 18 command variables. Loaded
dynamics/contact geometry are frozen within each 10ms interval. This remains
an unsaturated, fixed-contact model, not a PhysX or stability certificate.
"""
import time
import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse

from scripts.command_history_support_qp_v1 import CommandHistorySupportQP
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive,interval_constraints
from scripts.frame_point_kinematics_v1 import frame_point_kinematics
from scripts.implicit_support_prediction_v1 import predict_implicit_support


class PureControlIntervalQPBaseV1(CommandHistorySupportQP):
    def __init__(self,*args,substeps=16,minimum_root_height_m=0.,**kwargs):
        super().__init__(*args,**kwargs)
        if type(substeps) is not int or substeps<1 or not np.isfinite(minimum_root_height_m):
            raise ValueError('Explicit substeps and original root-height threshold required')
        self.substeps=substeps;self.minimum_root_height_m=float(minimum_root_height_m)

    def _make_solver(self):
        return osqp.OSQP()

    def solve(self,state,joint_reference,*,static_equilibrium=False):
        started=time.perf_counter();cfg=self.settings;interface=self.interface;dt=cfg.control_dt_s
        if cfg.force_distribution_weight or cfg.force_wrench_weight or cfg.nonfoot_clearance_hard:
            raise ValueError('Unsupported optional force/hard-clearance tasks must not be silently omitted')
        interface.bind_previous_command(state.previous_command_target)
        requested=np.asarray(joint_reference,dtype=float)
        if requested.shape!=(18,) or not np.isfinite(requested).all():raise ValueError('Finite reference required')
        if np.any(requested<interface.limits[:,0]-1e-6) or np.any(requested>interface.limits[:,1]+1e-6):
            raise ValueError('Reference outside original native limits')
        low_command,high_command=interface.admitted_target_bounds(state.joint_position,dt)
        requested=np.clip(requested,low_command,high_command)
        if static_equilibrium and np.max(abs(state.v))>1e-12:raise ValueError('Static fixture must have zero velocity')
        snap=self.model.snapshot(state.q,state.v)
        com_jac,com_bias=self.com_kinematics(state)
        self._static_fixture=bool(static_equilibrium)
        try:reference=self.reference(state,snap,requested)
        finally:self._static_fixture=False
        feet=np.flatnonzero(state.contacts);jac=snap['foot_jacobian'][feet].reshape(-1,24)
        jdot=(snap['foot_jacobian_dot'][feet]@state.v).reshape(-1)
        S=np.zeros((24,18));S[self.model.v_indices,np.arange(18)]=1.
        stages=condense_implicit_drive(snap['mass_matrix'],snap['nonlinear'],S,interface.kp,interface.kd,
            jac,jdot,state.v,state.joint_position,duration_s=dt,substeps=self.substeps,
            contact_velocity_decay_s=cfg.contact_velocity_decay_s)
        final=stages[-1];am=final['velocity_map']/dt;ab=(final['velocity_bias']-state.v)/dt
        H=np.zeros((18,18));g=np.zeros(18)
        def task(matrix,bias,target,weight):
            nonlocal H,g
            H+=weight*matrix.T@matrix;g+=weight*matrix.T@(bias-target)
        task(am,ab,np.zeros(24),cfg.acceleration_regularization)
        task(com_jac@am,com_jac@ab+com_bias,reference['com_acceleration'],cfg.com_weight)
        task(am[3:6],ab[3:6],reference['angular_acceleration_body'],cfg.orientation_weight)
        task(S.T@am,S.T@ab,reference['joint_acceleration'],cfg.posture_weight)
        for stage in stages:
            task(stage['torque_map'],stage['torque_bias'],np.zeros(18),cfg.torque_regularization/self.substeps)
            task(stage['force_map'],stage['force_bias'],np.zeros(len(stage['force_bias'])),cfg.force_regularization/self.substeps)
        foot_ids=[self.model.names.index(n) for n in self.model.feet]
        swing=reference.get('swing_foot_target_world')
        if cfg.swing_foot_weight>1e-7 and swing is not None:
            for foot in np.flatnonzero(~state.contacts):
                jf=snap['foot_jacobian'][foot]
                desired=cfg.swing_foot_kp_s2*(swing[foot]-snap['body_positions_world'][foot_ids[foot]])-cfg.swing_foot_kd_s*(jf@state.v)
                task(jf@am,jf@ab+snap['foot_jacobian_dot'][foot]@state.v,desired,cfg.swing_foot_weight)
        if False:  # Replaced by authored whole-body collision surfaces below.
            frames=frame_point_kinematics(self.model,state.q,state.v,('FL_thigh','FR_thigh','RL_thigh','RR_thigh'))
            ground=float(np.median(snap['body_positions_world'][np.asarray(foot_ids)[feet],2]))
            for frame in frames.values():
                z,vz=frame['position'][2],frame['velocity'][2];target_z=ground+cfg.nonfoot_clearance_m
                if z+dt*vz<=target_z+.12 or vz<-.5:
                    desired=np.clip(cfg.nonfoot_clearance_kp_s2*(target_z-z)-cfg.nonfoot_clearance_kd_s*vz,-80.,80.)
                    jf=frame['jacobian'][2:3]
                    task(jf@am,jf@ab+frame['jacobian_dot'][2:3]@state.v,np.array([desired]),cfg.nonfoot_clearance_weight)
        gaps, surface_jac = self.geometry.evaluate(state.q,jacobian=True)
        rates = surface_jac @ state.v
        lowest = {}
        for i, shape in enumerate(self.geometry.shapes):
            name=shape['body']
            if name not in lowest or gaps[i]<gaps[lowest[name]]:lowest[name]=i
        # Protective acceleration is a soft objective, not a demand that all
        # descending surfaces move upward. No current ground support is invented.
        for name,i in lowest.items():
            gap,rate=gaps[i],rates[i]
            if gap<.12 or rate<-.3:
                braking = max(0.,-rate)**2/(2.*max(gap-.005,.005))
                desired = np.clip(max(braking,80.*(.025-gap)-18.*rate),0.,80.)
                j=surface_jac[i:i+1]
                task(j@am,j@ab,np.array([desired]),80. if name.startswith('Head_') else 25.)
        interval=interval_constraints(stages,S,effort=interface.effort,velocity_limit=interface.velocity,
            joint_limits=interface.limits,friction=interface.friction,friction_facets=cfg.friction_facets)
        matrices=[interval['matrix'],np.eye(18)];lows=[interval['low'],low_command];highs=[interval['high'],high_command]
        if self.preserve_arm_command:
            matrices.append(np.eye(18)[12:]);lows.append(requested[12:]);highs.append(requested[12:])
        # Physical authored nonfoot surface, rather than a fixed base-height
        # or final-standing constraint. This local linearization is checked
        # again after manifold integration and is not a horizon certificate.
        displacement_map=np.zeros((24,18));displacement_bias=np.zeros(24)
        for stage in stages:
            displacement_map+=dt/self.substeps*stage['velocity_map']
            displacement_bias+=dt/self.substeps*stage['velocity_bias']
            ids=np.array(list(lowest.values()),dtype=int)
            matrices.append(surface_jac[ids]@displacement_map)
            lows.append(-gaps[ids]-surface_jac[ids]@displacement_bias)
            highs.append(np.full(len(ids),np.inf))
        if static_equilibrium:
            matrices.append(final['velocity_map']);lows.append(-final['velocity_bias']);highs.append(-final['velocity_bias'])
        A=np.vstack(matrices);low=np.concatenate(lows);high=np.concatenate(highs)
        plan=dict(accepted=False,contacts=state.contacts.copy(),reference=reference,native_execution=False,
            formulation='held-command-QP-with-all-implicit-substep-constraints',substeps=self.substeps,
            previous_command_target=interface.previous_command.tolist(),admitted_reference=requested.tolist(),
            decision_variables=18,constraint_rows=len(low))
        # Solve for displacement from the requested command. Original bounds
        # stay unchanged; row/objective scaling only improves numerics.
        scale=.5*(interface.limits[:,1]-interface.limits[:,0]);center=requested
        scaled=A*scale[None,:];row_scale=1./np.maximum(np.max(abs(scaled),axis=1),1e-12)
        hs=H*scale[:,None]*scale[None,:];gs=scale*(H@center+g)
        objective_scale=1./max(float(np.max(abs(np.diag(hs)))),1.)
        solver=self._make_solver()
        solver.setup(P=sparse.csc_matrix(np.triu(hs*objective_scale)),q=gs*objective_scale,
            A=sparse.csc_matrix(scaled*row_scale[:,None]),l=(low-A@center)*row_scale,u=(high-A@center)*row_scale,
            eps_abs=cfg.solver_eps,eps_rel=cfg.solver_eps,max_iter=cfg.solver_max_iterations,polish=True,verbose=False)
        result=solver.solve();plan.update(solver_status=result.info.status,solver_iterations=result.info.iter,
            solver_runtime_s=result.info.run_time,wall_runtime_s=time.perf_counter()-started)
        if result.info.status_val!=1 or result.x is None or not np.isfinite(result.x).all():
            plan['rejection']='Interval command QP did not return a solved finite result';return plan
        target=center+scale*result.x;response=A@target
        violation=float(max(0.,np.max(low-response),np.max(response-high)))
        prediction=predict_implicit_support(self.model,interface,state,target,duration_s=dt,substeps=self.substeps,
            contact_velocity_decay_s=cfg.contact_velocity_decay_s)
        predicted=prediction['state'];q=state.q.copy();min_height=float(q[2])
        min_gap=float(gaps.min())
        for stage in stages:
            v=stage['velocity_map']@target+stage['velocity_bias']
            q=pin.integrate(self.model.model,q,dt/self.substeps*v);min_height=min(min_height,float(q[2]))
            min_gap=min(min_gap,float(self.geometry.evaluate(q)[0].min()))
        residuals=dict(all_interval_constraints=violation,
            independent_velocity=float(np.max(abs(predicted.v-(final['velocity_map']@target+final['velocity_bias'])))),
            independent_joint_position=float(np.max(abs(predicted.joint_position-(final['joint_position_map']@target+final['joint_position_bias'])))),
            independent_manifold_position=float(np.max(abs(predicted.q-q))))
        residuals.update(prediction['diagnostics'])
        # Iteration count is a diagnostic, not a physical residual.
        residuals.pop('maximum_active_set_iterations',None)
        force=np.zeros((4,3));force[feet]=(final['force_map']@target+final['force_bias']).reshape(-1,3)
        plan.update(joint_position_target=target,normalized_action=interface.normalized_action(target),
            acceleration=(predicted.v-state.v)/dt,torque=final['torque_map']@target+final['torque_bias'],
            force_world=force,predicted_joint_position=predicted.joint_position,
            predicted_joint_velocity=predicted.joint_velocity,predicted_com_acceleration=com_jac@((predicted.v-state.v)/dt)+com_bias,
            predicted_minimum_root_height_m=min_height,residuals=residuals)
        plan['minimum_authored_nonfoot_gap_m']=min_gap
        plan['accepted']=max(residuals.values())<=cfg.residual_tolerance and min_gap>=-1e-6
        if not plan['accepted']:plan['rejection']='Independent interval dynamics, bounds or authored geometry check failed'
        return plan
