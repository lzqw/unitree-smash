"""Offline normal-impact ablation for a held position command.

Adds unilateral normal impulses for initially unobserved foot spheres that
would cross the plane within a microstep. Existing feet retain the old support
law. New feet have zero tangential force: this deliberately isolates normal
impact and is NOT a full frictional contact model or a deployable teacher.
All generalized velocities are solved together; no state clipping or fictitious
base force is used. No future observed state is consulted.
"""
import copy
import numpy as np
import pinocchio as pin
from scripts.implicit_support_prediction_v1 import implicit_velocity_step
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps


def center_normal_jacobians(model,q,v,spheres):
    pin.computeJointJacobians(model.model,model.data,q)
    pin.updateFramePlacements(model.model,model.data)
    rows=[]
    for sphere in spheres:
        fid=model.body_frames[sphere['body']];pose=model.data.oMf[fid]
        offset=pose.rotation@np.asarray(sphere['center_body_m'])
        jac=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        rows.append(jac[2]+offset[1]*jac[3]-offset[0]*jac[4])
    return np.asarray(rows)


def predict_sphere_normal_impact(model,interface,state,target,spheres,*,substeps=16,detect_impacts=True):
    if type(substeps) is not int or not 1<=substeps<=160:raise ValueError('Invalid substeps')
    target=np.asarray(target,float)
    if (target.shape!=(18,) or not np.isfinite(target).all() or
            np.any(target<interface.limits[:,0]-1e-6) or np.any(target>interface.limits[:,1]+1e-6)):
        raise ValueError('Invalid held command')
    if [s['body'] for s in spheres]!=model.feet:raise ValueError('Wrong sphere order')
    q=state.q.copy();v=state.v.copy();dt=.01/substeps
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.
    old=np.flatnonzero(state.contacts);normal=set();events=[]
    diagnostics=dict(dynamics_residual=0.,old_normal_violation_n=0.,old_friction_violation_n=0.,
        new_normal_violation_n=0.,foot_penetration_m=0.,joint_velocity_violation_rad_s=0.)
    for micro in range(substeps):
        snapshot=model.snapshot(q,v);gap=sphere_gaps(model,q,spheres)
        oldjac=snapshot['foot_jacobian'][old].reshape(-1,24)
        olddelta=dt*(-oldjac@v/.06-(snapshot['foot_jacobian_dot'][old]@v).reshape(-1))
        normaljac=center_normal_jacobians(model,q,v,spheres);banned=set()
        for active_iteration in range(10):
            ids=sorted(normal);jac=np.vstack([oldjac,normaljac[ids]])
            # Stop at the plane on approach; correct existing tiny penetration
            # over 10 ms. Never teleport q to the plane.
            normal_velocity=np.array([-gap[i]/(dt if gap[i]>0 else .01) for i in ids])
            delta=np.r_[olddelta,normal_velocity-normaljac[ids]@v]
            step=implicit_velocity_step(snapshot['mass_matrix'],v,snapshot['nonlinear'],selection,
                interface.kp,interface.kd,target-q[model.q_indices],interface.effort,dt,jac,delta)
            negative=[ids[i] for i,f in enumerate(step['force'][len(old)*3:]) if f < -1e-7]
            if negative:
                normal.difference_update(negative);banned.update(negative)
                events.append(dict(micro=micro,kind='normal_release',feet=negative));continue
            qtrial=pin.integrate(model.model,q,dt*step['velocity'])
            end_gap=sphere_gaps(model,qtrial,spheres)
            added=[i for i in range(4) if detect_impacts and i not in old and i not in normal and i not in banned
                   and end_gap[i]<-1e-7 and normaljac[i]@step['velocity']<0]
            if added:
                normal.update(added);events.append(dict(micro=micro,kind='normal_impact',feet=added,
                    gap_before_m=gap[added].tolist()));continue
            break
        else:raise ValueError('Contact active set did not converge')
        force=step['force'][:len(old)*3].reshape(-1,3)
        if len(force):
            diagnostics['old_normal_violation_n']=max(diagnostics['old_normal_violation_n'],float(np.max(-force[:,2])))
            diagnostics['old_friction_violation_n']=max(diagnostics['old_friction_violation_n'],
                float(np.max(np.linalg.norm(force[:,:2],axis=1)-interface.friction*force[:,2])))
        diagnostics['new_normal_violation_n']=max(diagnostics['new_normal_violation_n'],
            float(np.max(-step['force'][len(old)*3:],initial=0)))
        diagnostics['foot_penetration_m']=max(diagnostics['foot_penetration_m'],float(np.max(-end_gap)))
        diagnostics['dynamics_residual']=max(diagnostics['dynamics_residual'],step['residual'])
        q=qtrial;v=step['velocity']
        diagnostics['joint_velocity_violation_rad_s']=max(diagnostics['joint_velocity_violation_rad_s'],
            float(np.max(abs(v[model.v_indices])-interface.velocity)))
    following=copy.deepcopy(state);following.q=q;following.v=v
    following.joint_position=q[model.q_indices].copy();following.joint_velocity=v[model.v_indices].copy()
    following.root_position_world=q[:3].copy();following.root_rotation_world=pin.Quaternion(q[3:7]).matrix()
    following.contacts=state.contacts.copy();following.contacts[list(normal)]=True
    return dict(state=following,diagnostics=diagnostics,events=events,native_execution=False,
        valid_teacher=False,new_contacts_tangential_force='zero; normal-impact ablation only')
