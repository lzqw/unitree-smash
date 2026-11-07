"""Held-position, implicit-PD diagnostic on one frozen contact linearization.

Not a PhysX replica or a validated controller. Motor saturation is solved by
an active set; joint-limit/contact-mode violations are reported, never hidden
by clipping state or adding support. Mass, contact Jacobian and bias are frozen
within the short interval. Actual robot/hardware parameters are unchanged.
"""
import copy
import numpy as np
import pinocchio as pin
from scipy.linalg import lu_factor,lu_solve


def implicit_velocity_step(mass,velocity,bias,selection,kp,kd,target_error,
                           effort,dt,contact_jacobian,contact_velocity_delta,
                           *,factors=None):
    nv=len(velocity);na=selection.shape[1];nc=len(contact_jacobian)
    if (mass.shape!=(nv,nv) or selection.shape!=(nv,na) or
            any(np.shape(x)!=(na,) for x in (kp,kd,target_error,effort)) or
            np.shape(bias)!=(nv,) or contact_jacobian.shape!=(nc,nv) or
            np.shape(contact_velocity_delta)!=(nc,) or not dt>0):
        raise ValueError('Invalid implicit-step dimensions')
    if any(not np.isfinite(x).all() for x in (mass,velocity,bias,selection,kp,kd,target_error,
                                              effort,contact_jacobian,contact_velocity_delta)):
        raise ValueError('Nonfinite implicit-step inputs')
    if np.any(kp<=0) or np.any(kd<0) or np.any(effort<=0):raise ValueError('Invalid physical gains/effort')
    if factors is None:factors={}
    joint_velocity=selection.T@velocity
    drive_bias=kp*target_error
    drive_slope=kd+dt*kp
    saturated=np.zeros(na,dtype=np.int8);seen=set()
    for iteration in range(2*na+2):
        key=tuple(saturated.tolist())
        if key in seen:raise ValueError('Motor saturation active set cycled')
        seen.add(key);free=saturated==0
        if key not in factors:
            effective=mass+dt*(selection[:,free]*drive_slope[free])@selection[:,free].T
            matrix=np.block([[effective,-dt*contact_jacobian.T],
                [contact_jacobian,np.zeros((nc,nc))]])
            factors[key]=(matrix,lu_factor(matrix))
        matrix,factor=factors[key]
        drive=drive_bias.copy();drive[~free]=saturated[~free]*effort[~free]
        rhs=np.r_[mass@velocity-dt*bias+dt*selection@drive,
                  contact_jacobian@velocity+contact_velocity_delta]
        solved=lu_solve(factor,rhs)
        next_velocity=solved[:nv];force=solved[nv:]
        raw_torque=drive_bias-drive_slope*(selection.T@next_velocity)
        proposed=np.where(raw_torque>effort+1e-8,1,np.where(raw_torque < -effort-1e-8,-1,0)).astype(np.int8)
        if np.array_equal(proposed,saturated):
            torque=np.clip(raw_torque,-effort,effort)
            dynamics=mass@(next_velocity-velocity)/dt+bias-selection@torque-contact_jacobian.T@force
            contact=contact_jacobian@(next_velocity-velocity)-contact_velocity_delta
            residual=max(float(np.max(abs(dynamics))),float(np.max(abs(contact),initial=0.)))
            if not np.isfinite(solved).all() or residual>2e-5:
                raise ValueError('Implicit-step dynamics/contact residual failed')
            return dict(velocity=next_velocity,force=force,torque=torque,
                        active_set_iterations=iteration+1,residual=residual)
        saturated=proposed
    raise ValueError('Motor active set did not converge')


def predict_implicit_support(model,interface,state,target,*,duration_s=.01,
                             substeps=16,contact_velocity_decay_s=.06):
    target=np.asarray(target,dtype=np.float64)
    if target.shape!=(18,) or not np.isfinite(target).all():raise ValueError('Finite full command required')
    if np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6):
        raise ValueError('Command outside original position limits')
    if type(substeps) is not int or substeps<1 or not 0<duration_s<=.02:raise ValueError('Short diagnostic interval required')
    snapshot=model.snapshot(state.q,state.v);feet=np.flatnonzero(state.contacts)
    jac=snapshot['foot_jacobian'][feet].reshape(-1,24)
    jdot_v=(snapshot['foot_jacobian_dot'][feet]@state.v).reshape(-1)
    S=np.zeros((24,18));S[model.v_indices,np.arange(18)]=1.
    q=state.q.copy();v=state.v.copy();dt=duration_s/substeps;factors={}
    diagnostics=dict(dynamics_contact_residual_max=0.,joint_velocity_violation_rad_s=0.,
        joint_position_violation_rad=0.,normal_force_violation_n=0.,friction_violation_n=0.,
        motor_torque_violation_nm=0.,maximum_active_set_iterations=0)
    for _ in range(substeps):
        delta=dt*(-jac@v/contact_velocity_decay_s-jdot_v)
        step=implicit_velocity_step(snapshot['mass_matrix'],v,snapshot['nonlinear'],S,
            interface.kp,interface.kd,target-q[model.q_indices],interface.effort,dt,jac,delta,factors=factors)
        v=step['velocity'];q=pin.integrate(model.model,q,dt*v)
        force=step['force'].reshape(-1,3)
        diagnostics['dynamics_contact_residual_max']=max(diagnostics['dynamics_contact_residual_max'],step['residual'])
        diagnostics['maximum_active_set_iterations']=max(diagnostics['maximum_active_set_iterations'],step['active_set_iterations'])
        diagnostics['joint_velocity_violation_rad_s']=max(diagnostics['joint_velocity_violation_rad_s'],float(np.max(abs(v[model.v_indices])-interface.velocity)))
        diagnostics['joint_position_violation_rad']=max(diagnostics['joint_position_violation_rad'],float(np.max(interface.limits[:,0]-q[model.q_indices])),float(np.max(q[model.q_indices]-interface.limits[:,1])))
        diagnostics['motor_torque_violation_nm']=max(diagnostics['motor_torque_violation_nm'],float(np.max(abs(step['torque'])-interface.effort)))
        if len(feet):
            diagnostics['normal_force_violation_n']=max(diagnostics['normal_force_violation_n'],float(np.max(-force[:,2])))
            diagnostics['friction_violation_n']=max(diagnostics['friction_violation_n'],float(np.max(np.linalg.norm(force[:,:2],axis=1)-interface.friction*force[:,2])))
    following=copy.deepcopy(state);following.q=q;following.v=v
    following.joint_position=q[model.q_indices].copy();following.joint_velocity=v[model.v_indices].copy()
    following.root_position_world=q[:3].copy();following.root_rotation_world=pin.Quaternion(q[3:7]).matrix()
    return dict(state=following,diagnostics=diagnostics,duration_s=duration_s,substeps=substeps,
                contact_modes_changed=False,native_execution=False)
