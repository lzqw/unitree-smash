"""Rolling-contact velocity Jacobian at the authored sphere's bottom point.

The sphere centre and world-down radius offset differ from the foot-frame
origin. Forces and no-slip velocity constraints must use the same point.
This assumes the original horizontal ground, not arbitrary terrain.
"""
import numpy as np
import pinocchio as pin


def spherical_support_kinematics(loaded,q,v,spheres):
    if [x['body'] for x in spheres]!=loaded.feet:raise ValueError('Wrong authored sphere order')
    model,data=loaded.model,loaded.data
    pin.computeJointJacobians(model,data,q)
    pin.forwardKinematics(model,data,q,v,np.zeros(model.nv));pin.updateFramePlacements(model,data)
    jacobians=[];biases=[];positions=[]
    for sphere in spheres:
        fid=loaded.body_frames[sphere['body']];pose=data.oMf[fid]
        centre=pose.rotation@np.asarray(sphere['center_body_m'],float)
        offset=centre-np.array([0.,0.,sphere['radius_m']])
        jac=pin.getFrameJacobian(model,data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
        acceleration=pin.getFrameClassicalAcceleration(model,data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        omega=jac[3:]@v
        # The world-down radius vector is fixed, while the authored centre
        # offset rotates. This is d(J_contact*v)/dt for constant tangent v,
        # not the acceleration of one fixed material point on a spinning foot.
        bias=(acceleration.linear+np.cross(acceleration.angular,offset)
            +np.cross(omega,np.cross(omega,centre)))
        jacobians.append(jac[:3]-pin.skew(offset)@jac[3:])
        biases.append(bias.copy());positions.append(pose.translation+offset)
    return dict(jacobian=np.asarray(jacobians),jacobian_dot_velocity=np.asarray(biases),
        contact_positions_world=np.asarray(positions))
