"""Foot-contact prediction with coupled simulator speed-limit reactions.

Offline candidate, not an accepted MPC teacher. Optional authored sphere and
hip/mount responses remain diagnostic approximations, not complete collision
coverage, restitution, PhysX margin or compliance equivalence.
"""
import copy
import numpy as np
import pinocchio as pin
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step
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


def predict_speed_limited_spheres(model,interface,state,target,spheres,*,substeps=16,geometry=None,
                                 enforce_joint_stops=False,diagnostic_nonfoot_spheres=None,
                                 diagnostic_hip_mount_response=None,contact_max_iterations=400,
                                 audit_wrench=False):
    if type(audit_wrench) is not bool:raise ValueError('Explicit audit flag required')
    if type(substeps) is not int or not 1<=substeps<=160:raise ValueError('Invalid substeps')
    if type(contact_max_iterations) is not int or not 1<=contact_max_iterations<=4000:
        raise ValueError('Bounded contact iteration budget required')
    if diagnostic_hip_mount_response is not None and not enforce_joint_stops:
        raise ValueError('Hip/mount diagnostics require original joint stops')
    target=np.asarray(target,float)
    if (target.shape!=(18,) or not np.isfinite(target).all() or
            np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6)):
        raise ValueError('Invalid command')
    if [s['body'] for s in spheres]!=model.feet:raise ValueError('Wrong sphere order')
    extras=[] if diagnostic_nonfoot_spheres is None else copy.deepcopy(diagnostic_nonfoot_spheres)
    if any(s['body'] in model.feet or s['body'] not in model.body_frames for s in extras):
        raise ValueError('Invalid diagnostic nonfoot bodies')
    spheres=list(spheres)+extras
    contact_count=len(spheres)
    q=state.q.copy();v=state.v.copy();dt=.01/substeps;warm=None;last=None;events=[]
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.
    first_ground_violation=None
    wrench=(dict(contact_impulse_world_ns=np.zeros((contact_count,3)),
        contact_angular_impulse_com_world_nms=np.zeros((contact_count,3)),
        drive_torque_integral_nms=np.zeros(18),saturated_substeps=np.zeros(18,dtype=int)) if audit_wrench else None)
    diagnostics=dict(momentum_residual=0.,normal_complementarity=0.,friction_cone=0.,tangent_fixed_point=0.,
        normal_impulse_violation_ns=0.,max_contact_iterations=0,foot_penetration_m=0.,joint_velocity_violation_rad_s=0.,
        joint_position_violation_rad=0.,motor_torque_violation_nm=0.,nonfoot_ground_violation_m=0.,velocity_limit_active_substeps=0,
        maximum_velocity_limit_impulse_ns=0.,speed_constraint_residual=0.,
        nonfoot_normal_impulse_ns=0.,nonfoot_contact_substeps=0,
        position_limit_active_substeps=0,maximum_joint_stop_impulse_nms=0.,
        hip_mount_active_substeps=0,maximum_hip_mount_impulse_nms=0.)
    for micro in range(substeps):
        snapshot=model.snapshot(q,v);gap=sphere_gaps(model,q,spheres)
        jac=surface_jacobians(model,q,spheres).reshape(3*contact_count,24)
        normal_target=-gap/np.where(gap>0,dt,.01)
        # Ideal unilateral position stops coupled to contact, PD saturation and
        # original simulator speed constraints. Not calibrated PhysX compliance.
        bounds={} if not enforce_joint_stops else dict(
            position=q[model.q_indices],position_limits=interface.limits)
        step=position_limited_velocity_step if enforce_joint_stops else speed_limited_velocity_step
        arguments=(snapshot['mass_matrix'],v,snapshot['nonlinear'],selection,
            interface.kp,interface.kd,target-q[model.q_indices],interface.effort,dt,jac,normal_target,
            interface.friction,interface.velocity)
        if diagnostic_hip_mount_response is None:
            result=step(*arguments,warm=warm,max_iterations=contact_max_iterations,**bounds)
        else:
            result=diagnostic_hip_mount_response.step(*arguments,q[model.q_indices],interface.limits,
                warm=warm,max_iterations=contact_max_iterations)
        diagnostics['hip_mount_active_substeps']+=int(np.any(result.get('hip_mount_impulse',0.)!=0))
        diagnostics['maximum_hip_mount_impulse_nms']=max(diagnostics['maximum_hip_mount_impulse_nms'],
            float(np.max(abs(result.get('hip_mount_impulse',np.zeros(18))),initial=0.)))
        diagnostics['velocity_limit_active_substeps']+=int(result['velocity_limit_active'])
        diagnostics['maximum_velocity_limit_impulse_ns']=max(diagnostics['maximum_velocity_limit_impulse_ns'],float(np.max(abs(result['velocity_limit_impulse']),initial=0.)))
        diagnostics['speed_constraint_residual']=max(diagnostics['speed_constraint_residual'],result['speed_violation'],result['speed_fixed_point'])
        diagnostics['position_limit_active_substeps']+=int(result.get('position_limit_active',False))
        diagnostics['maximum_joint_stop_impulse_nms']=max(diagnostics['maximum_joint_stop_impulse_nms'],
            float(np.max(abs(result.get('position_limit_impulse',np.zeros(18))),initial=0.)))
        if wrench is not None:
            impulses=result['impulse'].reshape(contact_count,3)
            # Jacobians above leave frame placements at this pre-step q.
            points=np.array([model.data.oMf[model.body_frames[s['body']]].translation+
                model.data.oMf[model.body_frames[s['body']]].rotation@np.asarray(s['center_body_m'])-
                np.array([0.,0.,s['radius_m']]) for s in spheres])
            wrench['contact_impulse_world_ns']+=impulses
            wrench['contact_angular_impulse_com_world_nms']+=np.cross(points-snapshot['com_world'],impulses)
            wrench['drive_torque_integral_nms']+=result['torque']*dt
            wrench['saturated_substeps']+=(abs(result['torque'])>=interface.effort-1e-7)
        v=result['velocity'];warm=result['impulse'];q=pin.integrate(model.model,q,dt*v)
        normals=warm.reshape(contact_count,3)[:,2]
        diagnostics['normal_impulse_violation_ns']=max(diagnostics['normal_impulse_violation_ns'],float(np.max(-normals,initial=0.)))
        loaded=normals[:4]>1e-8
        diagnostics['nonfoot_normal_impulse_ns']+=float(np.sum(normals[4:]))
        diagnostics['nonfoot_contact_substeps']+=int(np.any(normals[4:]>1e-8))
        if last is None or not np.array_equal(loaded,last):
            events.append(dict(micro=micro,predicted_loaded_feet=loaded.tolist(),normal_impulse_ns=normals[:4].tolist()))
        last=loaded
        diagnostics['momentum_residual']=max(diagnostics['momentum_residual'],result['momentum_residual'])
        for key,value in result['contact']['residual'].items():diagnostics[key]=max(diagnostics[key],value)
        diagnostics['max_contact_iterations']=max(diagnostics['max_contact_iterations'],result['contact']['iterations'])
        diagnostics['foot_penetration_m']=max(diagnostics['foot_penetration_m'],float(np.max(-sphere_gaps(model,q,spheres[:4]))))
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
    following.previous_command_target=target.copy()
    return dict(state=following,diagnostics=diagnostics,events=events,native_execution=False,valid_teacher=False,
        ground_geometry_checked=geometry is not None,first_ground_violation=first_ground_violation,
        ideal_joint_position_stops=enforce_joint_stops,
        diagnostic_nonfoot_bodies=[s['body'] for s in extras],nonfoot_contact_not_admissible=True,
        diagnostic_hip_mount_response=diagnostic_hip_mount_response is not None,
        wrench_audit=wrench)
