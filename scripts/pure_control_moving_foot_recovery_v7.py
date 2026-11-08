"""World-frame position, velocity and acceleration references for airborne feet.

An airborne footprint moves with the whole-body COM. Damping it toward zero
world velocity would retract the legs while the body falls. This variant keeps
physical constraints, but supplies the derivatives of its moving foot goal.
"""
import numpy as np
import pinocchio as pin
from scripts.pure_control_self_collision_recovery_v5 import PureControlSelfCollisionRecoveryV5
from scripts.pure_control_interval_moving_foot_v3 import PureControlIntervalMovingFootV3


class PureControlMovingFootRecoveryV7(PureControlSelfCollisionRecoveryV5):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.previous_air_offset=None;self.previous_air_velocity=None;self.previous_air_time=None

    def solve(self,*args,**kwargs):
        plan=PureControlIntervalMovingFootV3.solve(self,*args,**kwargs)
        self.last_hip_plan=plan;return plan

    def reference(self,state,snapshot,joint_reference):
        result=super().reference(state,snapshot,joint_reference)
        result['swing_foot_velocity_world']=np.zeros((4,3))
        result['swing_foot_acceleration_world']=np.zeros((4,3))
        if state.contacts.any():
            self.previous_air_offset=None;self.previous_air_velocity=None;self.previous_air_time=None
            return result
        r=state.root_rotation_world;yaw=np.arctan2(r[1,0],r[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        qref=state.q.copy();qref[:3]=0.;qref[3:7]=pin.Quaternion(np.array([[c,-s,0],[s,c,0],[0,0,1.]])).coeffs()
        qref[self.model.q_indices]=joint_reference
        reference_snapshot=self.model.snapshot(qref,np.zeros(24))
        ids=[self.model.names.index(n) for n in self.model.feet]
        offset=reference_snapshot['body_positions_world'][ids]-reference_snapshot['com_world']
        offset[:,:2]+=self.reference_diagnostics['capture_offset']
        now=float(self.elapsed)
        offset_velocity=np.zeros((4,3));offset_acceleration=np.zeros((4,3))
        if self.previous_air_time is not None:
            dt=now-self.previous_air_time
            if 1e-6<dt<=.02:
                offset_velocity=(offset-self.previous_air_offset)/dt
                offset_acceleration=(offset_velocity-self.previous_air_velocity)/dt
        self.previous_air_time=now;self.previous_air_offset=offset.copy();self.previous_air_velocity=offset_velocity.copy()
        result['swing_foot_target_world']=snapshot['com_world']+offset
        result['swing_foot_velocity_world']=snapshot['com_velocity_world']+offset_velocity
        result['swing_foot_acceleration_world']=np.array([0.,0.,-9.81])+offset_acceleration
        self.reference_diagnostics.update(swing_feet=result['swing_foot_target_world'],
            foot_velocity_reference_world=result['swing_foot_velocity_world'],
            foot_acceleration_reference_world=result['swing_foot_acceleration_world'],
            zero_world_foot_velocity_assumption_removed=True)
        return result

    def command(self,state,phase,time_s):
        target,diagnostic=super().command(state,phase,time_s)
        bounded=np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        correction=float(np.max(abs(bounded-target)))
        if correction>1e-8:raise ValueError('Command outside physical bounds beyond roundoff')
        diagnostic.update(controller='pure_control_moving_foot_recovery_v7',
            moving_foot_reference=True,original_limit_roundoff_correction_rad=correction)
        return bounded,diagnostic
