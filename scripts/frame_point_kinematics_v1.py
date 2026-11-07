"""Copy world point kinematics at an explicit configuration.

Pinocchio Data is scratch space: a posture-reference FK must never become the
measured frame state used by a clearance task. These are link-origin points,
not collision-surface distances or a native contact model.
"""
import numpy as np
import pinocchio as pin


def frame_point_kinematics(loaded_model, q, v, names):
    model, data = loaded_model.model, loaded_model.data
    pin.computeJointJacobiansTimeVariation(model, data, q, v)
    pin.updateFramePlacements(model, data)
    result = {}
    for name in names:
        fid = loaded_model.body_frames[name]
        jac = pin.getFrameJacobian(model, data, fid,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
        world_dot = pin.getFrameJacobianTimeVariation(
            model, data, fid, pin.ReferenceFrame.WORLD).copy()
        position = data.oMf[fid].translation.copy()
        velocity = jac[:3] @ v
        # Translate WORLD spatial Jacobian derivative to the moving frame
        # origin; the point-velocity term is necessary as the origin moves.
        point_dot = (world_dot[:3] - np.cross(position, world_dot[3:].T).T
                     - np.cross(velocity, jac[3:].T).T)
        result[name] = dict(position=position, jacobian=jac[:3].copy(),
                            jacobian_dot=point_dot, velocity=velocity)
    return result
