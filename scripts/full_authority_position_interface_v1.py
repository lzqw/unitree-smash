"""Recovery position commands use physical joint ranges, not a residual box.

This changes only software command authority. Loaded motor torque, actual
velocity, stiffness/damping and geometry remain unchanged. A full-range target
is a request to the original drive, never a teleport or instantaneous torque
tracking guarantee. Predictors still have to enforce the real dynamics.
"""
import numpy as np
from scripts.whole_body_support_qp_v1 import PositionPDInterface


class FullAuthorityPositionPDInterfaceV1(PositionPDInterface):
    def __init__(self,source):
        super().__init__(source.limits,source.kp,source.kd,source.effort,source.velocity,source.friction)
        self.previous_command=None

    def bind_previous_command(self,command):
        command=np.asarray(command,float)
        if (command.shape!=(18,) or not np.isfinite(command).all()
            or np.any(command<self.limits[:,0]-1e-5) or np.any(command>self.limits[:,1]+1e-5)):
            raise ValueError('Actual previous full command must be finite and in physical range')
        self.previous_command=np.clip(command,self.limits[:,0],self.limits[:,1]).copy()

    def admitted_target_bounds(self,q,control_dt_s):
        if self.previous_command is None:raise ValueError('Actual command history required')
        if not np.isfinite(control_dt_s) or control_dt_s<=0:raise ValueError('Positive control period required')
        q=np.asarray(q,float)
        if q.shape!=(18,) or not np.isfinite(q).all():raise ValueError('Finite measured joint state required')
        return self.limits[:,0].copy(),self.limits[:,1].copy()

    def torque_bounds(self,q,v,control_dt_s=None):
        # Do not inherit the legacy q +/- velocity * dt software envelope.
        # These are instantaneous actuator bounds, not interval certificates.
        q=np.asarray(q,float);v=np.asarray(v,float)
        if q.shape!=(18,) or v.shape!=(18,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('Finite measured joint state required')
        low,high=self.admitted_target_bounds(q,.01 if control_dt_s is None else control_dt_s)
        return (np.clip(self.kp*(low-q)-self.kd*v,-self.effort,self.effort),
                np.clip(self.kp*(high-q)-self.kd*v,-self.effort,self.effort))


class FullAuthorityCommandArbiterV1:
    def __init__(self,limits):
        self.limits=np.asarray(limits,float).copy()
        if (self.limits.shape!=(18,2) or not np.isfinite(self.limits).all()
            or np.any(self.limits[:,0]>=self.limits[:,1])):raise ValueError('Original physical limits required')

    def select(self,student_target,controller_target,previous_final_target,*,owns_recovery,dt_s=.01):
        previous=np.asarray(previous_final_target,float)
        requested=np.asarray(controller_target if owns_recovery else student_target,float)
        if (previous.shape!=(18,) or requested.shape!=(18,) or not np.isfinite(previous).all()
            or not np.isfinite(requested).all() or dt_s!=.01):raise ValueError('Original clock and finite full commands required')
        target=np.clip(requested,self.limits[:,0],self.limits[:,1])
        return target,dict(owner='controller' if owns_recovery else 'frozen_prefix',
            requested_target_rad=requested.copy(),final_target_rad=target.copy(),
            command_mapping_delta_rad=target-requested,software_command_slew_limit=None,
            measured_joint_position_used_to_bound_target=False,student_residual_added_after_controller=False,
            native_motor_and_joint_limits_unchanged=True)
