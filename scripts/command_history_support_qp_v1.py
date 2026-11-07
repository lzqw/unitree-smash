"""Native joint-position interface with command-history arm slew.

Physical velocity is constrained by the QP next-state bounds and native motor.
It is not a target-error cap. Legs retain the native absolute position range;
arm software slew uses the previous final command, as ArmTargetRateV4 does.
"""
import numpy as np
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.continuous_support_qp_v1 import ContinuousSupportQP


class CommandHistoryPDInterface(PositionPDInterface):
    def __init__(self, source, control_dt_s, arm_target_rate_rad_s):
        super().__init__(source.limits,source.kp,source.kd,source.effort,source.velocity,source.friction)
        self.control_dt_s=float(control_dt_s)
        self.arm_rate=float(arm_target_rate_rad_s)
        if (not np.isfinite(self.control_dt_s) or self.control_dt_s<=0 or
                not np.isfinite(self.arm_rate) or not 0<self.arm_rate<=20):
            raise ValueError('Invalid declared command period or arm slew')
        self.previous_command=None
        self.arm_reference=None

    def bind_previous_command(self, command):
        a=np.asarray(command,dtype=np.float64)
        if a.shape!=(18,) or not np.isfinite(a).all():
            raise ValueError('Require actual previous final 18D command')
        if np.any(a<self.limits[:,0]-1e-5) or np.any(a>self.limits[:,1]+1e-5):
            raise ValueError('Previous command outside joint limits')
        self.previous_command=np.clip(a,self.limits[:,0],self.limits[:,1]).copy()

    def admitted_target_bounds(self,q,control_dt_s):
        if self.previous_command is None:
            raise ValueError('Command history is required, not inferred from measured joints')
        dt=float(control_dt_s)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid command period')
        low,high=self.limits[:,0].copy(),self.limits[:,1].copy()
        low[12:]=np.maximum(low[12:],self.previous_command[12:]-self.arm_rate*dt)
        high[12:]=np.minimum(high[12:],self.previous_command[12:]+self.arm_rate*dt)
        return low,high

    def torque_bounds(self,q,v,control_dt_s=None):
        low,high=self.admitted_target_bounds(q,self.control_dt_s if control_dt_s is None else control_dt_s)
        low=np.clip(self.kp*(low-q)-self.kd*v,-self.effort,self.effort)
        high=np.clip(self.kp*(high-q)-self.kd*v,-self.effort,self.effort)
        if self.arm_reference is not None:
            arm=np.clip(self.kp[12:]*(self.arm_reference-q[12:])-self.kd[12:]*v[12:],
                        -self.effort[12:],self.effort[12:])
            if np.any(arm<low[12:]-1e-8) or np.any(arm>high[12:]+1e-8):
                raise ValueError('Prescribed arm goal is inconsistent with command history')
            low[12:]=arm;high[12:]=arm
        return low,high

    def target_from_torque(self,torque,q,v):
        low,high=self.admitted_target_bounds(q,self.control_dt_s)
        target=np.clip(q+(torque+self.kd*v)/self.kp,low,high)
        if self.arm_reference is not None:target[12:]=self.arm_reference
        realized=np.clip(self.kp*(target-q)-self.kd*v,-self.effort,self.effort)
        return target,realized


class CommandHistorySupportQP(ContinuousSupportQP):
    def __init__(self,interface,settings,*,arm_target_rate_rad_s=5.,**kwargs):
        super().__init__(interface,settings,**kwargs)
        self.interface=CommandHistoryPDInterface(interface,settings.control_dt_s,arm_target_rate_rad_s)

    def reset(self):
        super().reset()
        if hasattr(self.interface, "previous_command"):
            self.interface.previous_command=None

    def reference_state(self):
        state=super().reference_state()
        state['previous_command']=(None if self.interface.previous_command is None
                                   else self.interface.previous_command.copy())
        return state

    def restore_reference_state(self,memory):
        super().restore_reference_state(memory)
        previous=memory.get('previous_command')
        self.interface.previous_command=None if previous is None else previous.copy()

    def solve(self,state,joint_reference,*,static_equilibrium=False):
        self.interface.bind_previous_command(state.previous_command_target)
        requested=np.asarray(joint_reference,dtype=np.float64)
        if requested.shape!=(18,) or not np.isfinite(requested).all():
            raise ValueError('Require a finite 18D reference')
        if np.any(requested<self.interface.limits[:,0]-1e-6) or np.any(requested>self.interface.limits[:,1]+1e-6):
            raise ValueError('Reference outside original joint limits')
        low,high=self.interface.admitted_target_bounds(state.joint_position,self.settings.control_dt_s)
        reference=np.clip(requested,low,high)
        plan=super().solve(state,reference,static_equilibrium=static_equilibrium)
        plan.update(command_interface='native-position-with-previous-command-arm-slew',
                    previous_command_target=self.interface.previous_command.tolist(),
                    admitted_reference=reference.tolist())
        return plan
