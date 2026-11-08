"""Feedback around the actual commands/states of a recorded recovery.

This is a diagnostic nominal reference, not automatically expert data. Its
source must retain native outcome, isolation and body-contact evidence. The
controller observes current state and previous dispatched command, not future
simulator states; the stored trajectory is a fixed offline reference.
"""
import numpy as np
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1


class RecordedTargetReferenceV1:
    def __init__(self,model,interface,commands):
        self.model=model;self.interface=interface;self.commands=np.asarray(commands,float).copy()
        if (self.commands.shape!=(520,18) or not np.isfinite(self.commands).all()
                or np.any(self.commands<interface.limits[:,0]-1e-7)
                or np.any(self.commands>interface.limits[:,1]+1e-7)):
            raise ValueError('Full original-clock finite recorded targets required')

    def action(self,offset,state,previous):
        step=offset-80
        if type(step) is not int or not 0<=step<520:raise ValueError('Original recorded clock required')
        target=np.clip(self.commands[step],self.interface.limits[:,0],self.interface.limits[:,1])
        target[12:]=np.clip(target[12:],previous[12:]-.01*self.interface.arm_rate,previous[12:]+.01*self.interface.arm_rate)
        return target,'executed_nominal_reference_candidate'


def executed_reference(model,interface,trace,gains):
    return LandingReferenceTrackingV1(RecordedTargetReferenceV1(model,interface,trace['requested_target_rad']),
        trace['q'],trace['v'],**gains)
