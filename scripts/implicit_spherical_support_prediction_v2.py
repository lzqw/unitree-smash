"""Independent held-command validation at the same authored sphere contacts."""
import copy
import numpy as np
import pinocchio as pin
from scripts.implicit_support_prediction_v1 import implicit_velocity_step
from scripts.spherical_support_kinematics_v1 import spherical_support_kinematics


def predict_implicit_support_spherical(model,interface,state,target,*,spheres,duration_s=.01,
                             substeps=16,contact_velocity_decay_s=.06):
    target=np.asarray(target,dtype=np.float64)
    if target.shape!=(18,) or not np.isfinite(target).all():raise ValueError('Finite full command required')
    if np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6):
        raise ValueError('Command outside original position limits')
    if type(substeps) is not int or substeps<1 or not 0<duration_s<=.02:raise ValueError('Short diagnostic interval required')
    snapshot=model.snapshot(state.q,state.v);feet=np.flatnonzero(state.contacts)
    contact=spherical_support_kinematics(model,state.q,state.v,spheres)
    jac=contact['jacobian'][feet].reshape(-1,24)
    jdot_v=contact['jacobian_dot_velocity'][feet].reshape(-1)
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
