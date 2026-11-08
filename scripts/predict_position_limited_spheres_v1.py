"""Candidate foot prediction with coupled PD, friction, speed and position stops.

Original motor limits and body geometry remain unchanged. Position stops use
zero restitution and first-order unilateral nonpenetration through the coupled
mass matrix; no post-integration position/velocity clipping is performed.
Nonfoot/self-collision response and exact PhysX equivalence remain unmodeled.
"""
import copy
import numpy as np
import pinocchio as pin
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step


def surface_jacobians(model,q,spheres):
    pin.computeJointJacobians(model.model,model.data,q);pin.updateFramePlacements(model.model,model.data)
    rows=[]
    for sphere in spheres:
        fid=model.body_frames[sphere['body']];pose=model.data.oMf[fid]
        offset=pose.rotation@np.asarray(sphere['center_body_m'])-np.array([0.,0.,sphere['radius_m']])
        jac=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        rows.append(jac[:3]-pin.skew(offset)@jac[3:])
    return np.asarray(rows)


def predict_position_limited_spheres(model,interface,state,target,spheres,*,substeps=16,geometry=None):
    if type(substeps) is not int or not 1<=substeps<=160:raise ValueError('Invalid substeps')
    target=np.asarray(target,float)
    if (target.shape!=(18,) or not np.isfinite(target).all() or
            np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6)):
        raise ValueError('Invalid command')
    if [s['body'] for s in spheres]!=model.feet:raise ValueError('Wrong sphere order')
    q=state.q.copy();v=state.v.copy();dt=.01/substeps;warm=None;last=None;events=[]
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.
    first_ground_violation=None
    diagnostics=dict(momentum_residual=0.,normal_complementarity=0.,friction_cone=0.,tangent_fixed_point=0.,
        max_contact_iterations=0,foot_penetration_m=0.,joint_velocity_violation_rad_s=0.,
        joint_position_violation_rad=0.,motor_torque_violation_nm=0.,nonfoot_ground_violation_m=0.,velocity_limit_active_substeps=0,
        maximum_velocity_limit_impulse_ns=0.,speed_constraint_residual=0.,
        position_limit_active_substeps=0,maximum_joint_stop_impulse_nms=0.,joint_box_residual=0.)
    for micro in range(substeps):
        snapshot=model.snapshot(q,v);gap=sphere_gaps(model,q,spheres)
        jac=surface_jacobians(model,q,spheres).reshape(12,24)
        normal_target=-gap/np.where(gap>0,dt,.01)
        result=position_limited_velocity_step(snapshot['mass_matrix'],v,snapshot['nonlinear'],selection,
            interface.kp,interface.kd,target-q[model.q_indices],interface.effort,dt,jac,normal_target,
            interface.friction,interface.velocity,q[model.q_indices],interface.limits,warm=warm)
        diagnostics['velocity_limit_active_substeps']+=int(result['velocity_limit_active'])
        diagnostics['maximum_velocity_limit_impulse_ns']=max(diagnostics['maximum_velocity_limit_impulse_ns'],float(np.max(abs(result['velocity_limit_impulse']),initial=0.)))
        diagnostics['speed_constraint_residual']=max(diagnostics['speed_constraint_residual'],result['speed_violation'],result['speed_fixed_point'])
        diagnostics['position_limit_active_substeps']+=int(result['position_limit_active'])
        diagnostics['maximum_joint_stop_impulse_nms']=max(diagnostics['maximum_joint_stop_impulse_nms'],float(np.max(abs(result['position_limit_impulse']),initial=0.)))
        diagnostics['joint_box_residual']=max(diagnostics['joint_box_residual'],result['joint_box_violation'],result['joint_box_fixed_point'])
        v=result['velocity'];warm=result['impulse'];q=pin.integrate(model.model,q,dt*v)
        loaded=warm.reshape(4,3)[:,2]>1e-8
        if last is None or not np.array_equal(loaded,last):
            events.append(dict(micro=micro,predicted_loaded_feet=loaded.tolist(),normal_impulse_ns=warm[2::3].tolist()))
        last=loaded
        diagnostics['momentum_residual']=max(diagnostics['momentum_residual'],result['momentum_residual'])
        for key,value in result['contact']['residual'].items():diagnostics[key]=max(diagnostics[key],value)
        diagnostics['max_contact_iterations']=max(diagnostics['max_contact_iterations'],result['contact']['iterations'])
        diagnostics['foot_penetration_m']=max(diagnostics['foot_penetration_m'],float(np.max(-sphere_gaps(model,q,spheres))))
        diagnostics['joint_velocity_violation_rad_s']=max(diagnostics['joint_velocity_violation_rad_s'],float(np.max(abs(v[model.v_indices])-interface.velocity)))
        diagnostics['joint_position_violation_rad']=max(diagnostics['joint_position_violation_rad'],
            float(np.max(interface.limits[:,0]-q[model.q_indices])),float(np.max(q[model.q_indices]-interface.limits[:,1])))
        diagnostics['motor_torque_violation_nm']=max(diagnostics['motor_torque_violation_nm'],float(np.max(abs(result['torque'])-interface.effort)))
        if geometry is not None:
            distances,_=geometry.evaluate(q);minimum=int(np.argmin(distances))
            diagnostics['nonfoot_ground_violation_m']=max(diagnostics['nonfoot_ground_violation_m'],-float(distances[minimum]))
            if first_ground_violation is None and distances[minimum]<-1e-6:
                first_ground_violation=dict(micro=micro,body=geometry.shapes[minimum]['body'],gap_m=float(distances[minimum]))
    following=copy.deepcopy(state);following.q=q;following.v=v
    following.joint_position=q[model.q_indices].copy();following.joint_velocity=v[model.v_indices].copy()
    following.root_position_world=q[:3].copy();following.root_rotation_world=pin.Quaternion(q[3:7]).matrix()
    following.contacts=last
    return dict(state=following,diagnostics=diagnostics,events=events,native_execution=False,valid_teacher=False,
        ground_geometry_checked=geometry is not None,first_ground_violation=first_ground_violation)
