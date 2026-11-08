"""State-feedback foot/body reference with the original position actuator map.

Measured supporting feet provide instantaneous anchors. A bounded virtual
body displacement is converted to joint targets by IK; swing feet approach
the declared landing footprint. Coupled rollout, not virtual IK, determines
whether this actually produces the intended support and body response.
"""
import numpy as np
import pinocchio as pin

from scripts.loaded_foot_recovery_reference_v2 import (
    LoadedFootRecoveryReferenceV2, PARAMETER_NAMES as BASE_NAMES,
    LOWER as BASE_LOWER,UPPER as BASE_UPPER,NOMINAL as BASE_NOMINAL,
)

PARAMETER_NAMES=BASE_NAMES+['body_position_gain','body_velocity_damping_s','body_rotation_gain']
LOWER=np.r_[BASE_LOWER,.05,.01,.05]
UPPER=np.r_[BASE_UPPER,.5,.12,.5]
NOMINAL=np.r_[BASE_NOMINAL,.2,.04,.2]


class FeedbackFootRecoveryReferenceV3(LoadedFootRecoveryReferenceV2):
    def __init__(self,model,interface,initial,spheres,parameters,*,control_tick):
        p=np.asarray(parameters,float)
        if p.shape!=(19,) or not np.isfinite(p).all() or np.any(p<LOWER) or np.any(p>UPPER):
            raise ValueError('Declared bounded state-feedback parameters required')
        super().__init__(model,interface,initial,spheres,p[:16],control_tick=control_tick)
        self.parameters=p.copy()
        rotation=initial.root_rotation_world
        yaw=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        self.upright=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        self.rotation_step=pin.log3(rotation.T@self.upright)
        self.final_root=initial.q[:3]+self.upright@np.r_[p[2:4],0.]
        self.final_root[2]=p[1]
        self.initial_feet=self.centers(initial.q)
        self.last_diagnostic=None

    def action(self,offset,state,previous):
        nominal,mode=super().action(offset,state,previous)
        tick=offset-80+280
        if tick==self.tick:return nominal,mode
        p=self.parameters;t=.01*(tick-self.tick)
        phase=float(np.clip(t/p[0],0.,1.));blend=phase*phase*(3.-2.*phase)
        root_goal=(1.-blend)*self.initial.q[:3]+blend*self.final_root
        rotation_goal=self.initial.root_rotation_world@pin.exp3(blend*self.rotation_step)
        rotation=state.root_rotation_world
        velocity_world=rotation@state.v[:3]
        shift=np.clip(p[16]*(root_goal-state.q[:3])-p[17]*velocity_world,-.03,.03)
        turn=np.clip(p[18]*pin.log3(rotation.T@rotation_goal)-p[17]*state.v[3:6],-.08,.08)
        virtual=state.q.copy();virtual[:3]+=shift
        virtual[3:7]=pin.Quaternion(rotation@pin.exp3(turn)).coeffs()
        virtual[self.model.q_indices[12:]]=nominal[12:]
        # These are actual current contacts; no swing foot is granted support.
        desired=(1.-blend)*self.initial_feet+blend*self.foot_center_targets_world
        current=self.centers(state.q)
        desired[np.asarray(state.contacts,bool)]=current[np.asarray(state.contacts,bool)]
        for _ in range(8):
            centers,jacobian=self.centers(virtual,jacobian=True)
            error=(desired-centers).reshape(-1)
            if np.max(abs(error))<1e-5:break
            legs=jacobian.reshape(12,self.model.model.nv)[:,self.model.v_indices[:12]]
            delta=legs.T@np.linalg.solve(legs@legs.T+.003**2*np.eye(12),error)
            virtual[self.model.q_indices[:12]]=np.clip(
                virtual[self.model.q_indices[:12]]+np.clip(delta,-.18,.18),
                self.interface.limits[:12,0],self.interface.limits[:12,1])
        target=nominal.copy()
        target[:12]=virtual[self.model.q_indices[:12]]+p[15]*np.exp(-t/p[14])*self.initial_command_preload[:12]
        target=np.clip(target,self.interface.limits[:,0],self.interface.limits[:,1])
        handover=float(np.clip(t/.08,0.,1.))
        target=nominal+handover*(target-nominal)
        self.last_diagnostic=dict(root_goal=root_goal.tolist(),virtual_translation=shift.tolist(),
            virtual_rotation=turn.tolist(),actual_contacts=np.asarray(state.contacts,bool).tolist(),
            maximum_ik_residual_m=float(np.linalg.norm(desired-self.centers(virtual),axis=1).max()))
        return target,'measured_support_body_feedback'


class FeedbackFootRecoveryPolicyV3:
    def __init__(self,reference):
        self.reference=reference;self.model=reference.model;self.interface=reference.interface

    def action(self,offset,state,previous):
        return self.reference.action(offset,state,previous)
