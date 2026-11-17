"""Experimental implicit PD prediction with contact release and Coulomb slip.

Local active-set approximation, NOT PhysX, SAP, or a safety certificate. Normal
contact uses a velocity/gap complementarity condition. Sliding friction opposes
the solved tangential velocity; sticking uses the declared velocity relaxation.
Cycles/nonconvergence are explicit failures, never a bilateral fallback.
Only initially observed candidate feet can contact during this short diagnostic.
"""
import copy
import numpy as np
import pinocchio as pin


def unilateral_velocity_step(mass, velocity, bias, selection, kp, kd,
                             target_error, effort, dt, jacobian,
                             desired_contact_velocity, friction, *, max_iterations=80):
    nv, na = selection.shape
    nc = len(jacobian)
    if (mass.shape != (nv,nv) or velocity.shape != (nv,) or bias.shape != (nv,)
            or jacobian.shape != (nc,3,nv) or desired_contact_velocity.shape != (nc,3)
            or any(np.shape(x) != (na,) for x in (kp,kd,target_error,effort))
            or not 0 < dt <= .02 or not np.isfinite(friction) or friction < 0):
        raise ValueError('Invalid contact/implicit motor dimensions or parameters')
    if any(not np.isfinite(x).all() for x in (mass,velocity,bias,selection,kp,kd,
            target_error,effort,jacobian,desired_contact_velocity)):
        raise ValueError('Nonfinite predictor input')
    if np.any(kp<=0) or np.any(kd<0) or np.any(effort<=0):
        raise ValueError('Invalid motor parameters')
    modes = np.ones(nc, dtype=np.int8)  # 0 released, 1 sticking, 2 sliding
    directions = np.zeros((nc,2))
    saturated = np.zeros(na, dtype=np.int8)
    drive = kp*target_error
    slope = kd+dt*kp
    force_tol, velocity_tol = 2e-5, 2e-6
    for iteration in range(max_iterations):
        free = saturated == 0
        effective = mass + dt*(selection[:,free]*slope[free])@selection[:,free].T
        motor_rhs = drive.copy()
        motor_rhs[~free] = saturated[~free]*effort[~free]
        rhs = mass@velocity + dt*(selection@motor_rhs-bias)
        columns, constraints, contact_rhs, slots = [], [], [], []
        for i, mode in enumerate(modes):
            if mode == 1:
                start = len(columns)
                columns.extend(jacobian[i])
                constraints.extend(jacobian[i])
                contact_rhs.extend(desired_contact_velocity[i])
                slots.append((i, start, 3))
            elif mode == 2:
                start = len(columns)
                direction = np.r_[-friction*directions[i],1.]
                columns.append(jacobian[i].T@direction)
                constraints.append(jacobian[i,2])
                contact_rhs.append(desired_contact_velocity[i,2])
                slots.append((i,start,1))
        B = np.asarray(columns).reshape(-1,nv).T
        C = np.asarray(constraints).reshape(-1,nv)
        K = np.block([[effective,-dt*B],[C,np.zeros((len(C),len(C)))]])
        rhs_full = np.r_[rhs,contact_rhs]
        solved = np.linalg.solve(K,rhs_full)
        vn = solved[:nv]
        forces = np.zeros((nc,3))
        for i,start,count in slots:
            f = solved[nv+start:nv+start+count]
            forces[i] = f if count == 3 else np.r_[-friction*directions[i],1.]*f[0]
        raw_torque = drive-slope*(selection.T@vn)
        new_sat = np.where(raw_torque>effort+1e-8,1,
                          np.where(raw_torque < -effort-1e-8,-1,0)).astype(np.int8)
        if not np.array_equal(new_sat,saturated):
            saturated = new_sat
            continue
        cv = np.einsum('cij,j->ci',jacobian,vn)
        changed = False
        for i, mode in enumerate(modes):
            if mode and forces[i,2] < -force_tol:
                modes[i] = 0
                changed = True
            elif mode == 0 and cv[i,2] < desired_contact_velocity[i,2]-velocity_tol:
                modes[i] = 1
                changed = True
            elif mode == 1 and np.linalg.norm(forces[i,:2]) > friction*max(0.,forces[i,2])+force_tol:
                modes[i] = 2
                # Use the force required to prevent slip for the initial
                # direction. Subsequent iterations use solved velocity.
                directions[i] = -forces[i,:2]/np.linalg.norm(forces[i,:2])
                changed = True
            elif mode == 2:
                speed = np.linalg.norm(cv[i,:2])
                if speed < 1e-8:
                    modes[i] = 1
                    changed = True
                else:
                    desired_direction = cv[i,:2]/speed
                    error = np.linalg.norm(directions[i]-desired_direction)
                    if error > 1e-4:
                        mixed = directions[i]+desired_direction
                        directions[i] = (mixed/np.linalg.norm(mixed)
                                         if np.linalg.norm(mixed)>1e-8 else desired_direction)
                        changed = True
        if changed:
            continue
        torque = np.clip(raw_torque,-effort,effort)
        reaction = np.einsum('cij,ci->j',jacobian,forces)
        dynamic_error = mass@(vn-velocity)/dt+bias-selection@torque-reaction
        normal_slack = cv[:,2]-desired_contact_velocity[:,2]
        residual = float(np.max(abs(dynamic_error),initial=0.))
        normal_violation = float(np.max(-forces[:,2],initial=0.))
        friction_violation = float(np.max(np.linalg.norm(forces[:,:2],axis=1)-friction*forces[:,2],initial=0.))
        separation_violation = float(np.max(-normal_slack,initial=0.))
        complementarity = float(np.max(abs(forces[:,2]*normal_slack),initial=0.))
        if (not np.isfinite(solved).all() or residual>2e-5 or normal_violation>force_tol
                or friction_violation>force_tol or separation_violation>velocity_tol
                or complementarity>2e-4):
            raise ValueError('Contact/motor consistency check failed')
        return dict(velocity=vn,force=forces,torque=torque,modes=modes.copy(),
            iterations=iteration+1,residual=residual,normal_complementarity=complementarity,
            normal_force_violation_n=normal_violation,friction_violation_n=friction_violation)
    raise ValueError('Contact/motor active set did not converge; no fallback applied')


def predict_unilateral_support(model, interface, state, target, *, duration_s=.01,
                               substeps=16, contact_velocity_decay_s=.06):
    target = np.asarray(target,dtype=np.float64)
    if target.shape != (18,) or not np.isfinite(target).all():
        raise ValueError('Finite full command required')
    if np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6):
        raise ValueError('Target outside original limits')
    if type(substeps) is not int or substeps<1 or not 0<duration_s<=.02:
        raise ValueError('Short diagnostic interval required')
    q,v = state.q.copy(),state.v.copy()
    feet = np.flatnonzero(state.contacts)
    S = np.zeros((24,18));S[model.v_indices,np.arange(18)]=1.
    feet_ids = [model.names.index(n) for n in model.feet]
    initial = model.snapshot(q,v)
    # Local contact heights come only from currently observed contacts. No
    # future force flags, case labels, or actual next states enter prediction.
    heights = initial['body_positions_world'][feet_ids][feet,2].copy()
    dt = duration_s/substeps
    diag = dict(dynamics_residual_max=0.,normal_force_violation_n=0.,friction_violation_n=0.,
        joint_velocity_violation_rad_s=0.,joint_position_violation_rad=0.,
        released_foot_substeps=0,sliding_foot_substeps=0,maximum_iterations=0,
        motor_torque_violation_nm=0.)
    last = None
    for _ in range(substeps):
        snap = model.snapshot(q,v)
        jac = snap['foot_jacobian'][feet]
        jdot_v = snap['foot_jacobian_dot'][feet]@v
        gap = snap['body_positions_world'][feet_ids][feet,2]-heights
        desired = np.einsum('cij,j->ci',jac,v)*(1.-dt/contact_velocity_decay_s)-dt*jdot_v
        desired[:,2] = -gap/dt-dt*jdot_v[:,2]
        step = unilateral_velocity_step(snap['mass_matrix'],v,snap['nonlinear'],S,
            interface.kp,interface.kd,target-q[model.q_indices],interface.effort,
            dt,jac,desired,interface.friction)
        v = step['velocity'];q = pin.integrate(model.model,q,dt*v)
        diag['motor_torque_violation_nm']=max(diag['motor_torque_violation_nm'],
            float(np.max(abs(step['torque'])-interface.effort)))
        diag['dynamics_residual_max']=max(diag['dynamics_residual_max'],step['residual'])
        for key in ['normal_force_violation_n','friction_violation_n']:
            diag[key]=max(diag[key],step[key])
        diag['released_foot_substeps']+=int(np.sum(step['modes']==0))
        diag['sliding_foot_substeps']+=int(np.sum(step['modes']==2))
        diag['maximum_iterations']=max(diag['maximum_iterations'],step['iterations'])
        diag['joint_velocity_violation_rad_s']=max(diag['joint_velocity_violation_rad_s'],
            float(np.max(abs(v[model.v_indices])-interface.velocity)))
        diag['joint_position_violation_rad']=max(diag['joint_position_violation_rad'],
            float(np.max(interface.limits[:,0]-q[model.q_indices])),
            float(np.max(q[model.q_indices]-interface.limits[:,1])))
        last=step
    following=copy.deepcopy(state);following.q=q;following.v=v
    following.joint_position=q[model.q_indices].copy();following.joint_velocity=v[model.v_indices].copy()
    following.root_position_world=q[:3].copy();following.root_rotation_world=pin.Quaternion(q[3:7]).matrix()
    following.contacts[:]=False
    following.contacts[feet]=last['modes']!=0
    forces = np.zeros((4,3))
    forces[feet] = last['force']
    return dict(state=following,diagnostics=diag,native_execution=False,
        foot_forces_world_n=forces,
        new_contact_candidates_added=False,impact_or_compliance_calibrated=False)
