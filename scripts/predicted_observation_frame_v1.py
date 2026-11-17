"""Reconstruct the native 69D current-frame convention from predicted state."""
import numpy as np
import pinocchio as pin


def predicted_frame(model,state,default,ball_position,ball_velocity,ball_present=True):
    pin.computeJointJacobians(model.model,model.data,state.q)
    pin.updateFramePlacements(model.model,model.data)
    fid=model.body_frames['Racket'];pose=model.data.oMf[fid]
    jac=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    twist=jac@state.v;R=state.root_rotation_world
    offset=R.T@(pose.translation-state.root_position_world)
    tool_velocity=R.T@twist[:3]-state.v[:3]-np.cross(state.v[3:6],offset)
    tool_angular=R.T@twist[3:]-state.v[3:6]
    ball=R.T@(np.asarray(ball_position)-state.root_position_world)
    ball_v=R.T@ball_velocity-state.v[:3]-np.cross(state.v[3:6],ball)
    frame=np.r_[state.v[:6],R.T@np.array([0.,0.,-1.]),
        state.joint_position-default,state.joint_velocity,offset,R.T@pose.rotation[:,0],
        tool_velocity,tool_angular,state.contacts.astype(float),
        ball*bool(ball_present),ball_v*bool(ball_present),float(ball_present),1.]
    if frame.shape!=(69,) or not np.isfinite(frame).all():raise ValueError('Invalid predicted frame')
    return frame


def encode_final_command(target,limits):
    target=np.asarray(target,dtype=float);limits=np.asarray(limits,dtype=float)
    if target.shape!=(18,) or limits.shape!=(18,2) or not np.isfinite(target).all():
        raise ValueError('Finite final18 target required')
    action=2*(target-limits[:,0])/(limits[:,1]-limits[:,0])-1
    if not np.isfinite(action).all() or np.any(np.abs(action)>1+1e-7):
        raise ValueError('Final command outside original position limits')
    return action
