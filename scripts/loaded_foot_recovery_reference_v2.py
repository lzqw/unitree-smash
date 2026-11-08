"""Preserve measured command preload during a proposed foot-anchored recovery.

Target-minus-measured-position is an actuator command history, not a measured
support force. Its decay is optimized and checked by the coupled dynamics;
no extra force, changed gain, or modified torque limit is introduced.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from scripts.foot_anchored_recovery_reference_v1 import (
    FootAnchoredRecoveryReferenceV1, FootAnchoredRecoveryFeedbackV1,
    PARAMETER_NAMES as BASE_NAMES, LOWER as BASE_LOWER, UPPER as BASE_UPPER,
    NOMINAL as BASE_NOMINAL,
)

PARAMETER_NAMES=BASE_NAMES+['command_preload_decay_s','command_preload_fraction']
LOWER=np.r_[BASE_LOWER,.08,0.]
UPPER=np.r_[BASE_UPPER,1.2,1.]
NOMINAL=np.r_[BASE_NOMINAL,.5,1.]


class LoadedFootRecoveryReferenceV2(FootAnchoredRecoveryReferenceV1):
    def __init__(self,model,interface,initial,spheres,parameters,*,control_tick):
        p=np.asarray(parameters,float)
        if p.shape!=(16,) or not np.isfinite(p).all() or np.any(p<LOWER) or np.any(p>UPPER):
            raise ValueError('Bounded 16-parameter command preload reference required')
        super().__init__(model,interface,initial,spheres,p[:14],control_tick=control_tick)
        self.parameters=p.copy()
        # Use native model joint mapping, never assume q[7:] is action order.
        self.initial_command_preload=initial.previous_command_target-initial.joint_position
        times=self.curve.x.copy();targets=self.curve(times)
        targets[1:,:12]+=p[15]*np.exp(-times[1:,None]/p[14])*self.initial_command_preload[None,:12]
        targets=np.clip(targets,interface.limits[:,0],interface.limits[:,1])
        targets[0]=initial.previous_command_target.copy()
        self.curve=PchipInterpolator(times,targets,axis=0,extrapolate=False)


class LoadedFootRecoveryFeedbackV2(FootAnchoredRecoveryFeedbackV1):
    def action(self,offset,state,previous):
        tick=offset-80+280
        if tick==self.reference.tick:return self.reference.action(offset,state,previous)
        target,mode=self.feedback.action(offset,state,previous)
        nominal,_=self.reference.action(offset,state,previous)
        # Ramp from this actual handover, not the older 2.8-second reference.
        weight=float(np.clip((tick-self.reference.tick)*.01/.2,0.,1.))
        return nominal+weight*(target-nominal),'loaded_foot_reference_with_feedback'
