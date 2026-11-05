"""Serializable assistance settings shared by TRAIN and independent replay."""
import math
import torch

from .reactive_landing_v1 import ReactiveLandingV1


RECOVERY_FIELDS={
    'recovery_local_time_s':(3,1.15,2.2),
    'recovery_thigh_bias_rad':(32,.05,.45),
    'recovery_front_back_thigh_bias_rad':(33,-.15,.15),
    'recovery_calf_bias_rad':(34,-.45,.05),
    'recovery_roll_feedback_gain':(35,-1.,0.),
    'recovery_actor_gain':(36,.2,1.2),
    'recovery_blend_s':(None,0.,.25),
    # A causal, optional sign-aware roll correction.  The legacy path keeps
    # the historical symmetric correction when this is zero.
    'recovery_side_aware_roll_feedback':(None,0.,1.),
}


def apply_recovery_assistance(reference,settings):
    if not settings:return
    if not isinstance(settings,dict) or set(settings)-set(RECOVERY_FIELDS):
        raise ValueError('Unknown recovery assistance field')
    if len(reference.base)<37:raise ValueError('Require recovery37 parameter layout')
    for name,value in settings.items():
        _,low,high=RECOVERY_FIELDS[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid recovery assistance: '+name)
    # A CPU .to(device) may alias a loaded checkpoint. Do not mutate it.
    reference.base=reference.base.clone()
    for name,value in settings.items():
        index,_,_=RECOVERY_FIELDS[name]
        if name == 'recovery_side_aware_roll_feedback':
            reference.teacher.standing.side_aware_roll_feedback = bool(float(value) > .5)
        elif index is None:reference.teacher.leg_recovery_blend.seconds=float(value)
        else:reference.base[index]=float(value)
    reference.parameters.copy_(reference.base[None].expand_as(reference.parameters))


class LandingAssistedController:
    def __init__(self,source,count,device,config,asset_sha256,enabled_cases=None):
        self.source=source;self.landing=ReactiveLandingV1(count,device,config,asset_sha256)
        self.enabled_cases=None if enabled_cases is None else set(int(x) for x in enabled_cases)

    @property
    def reference_controller(self):return self.source.reference_controller
    @property
    def logs(self):return self.source.logs
    @logs.setter
    def logs(self,value):self.source.logs=value

    def reset(self,ids=None):
        self.source.reset(ids);self.landing.reset(ids)

    def action(self,observation,default,limits):
        base=self.source.action(observation,default,limits)
        proposed=self.landing.action(base,observation['policy'],default,limits)
        if self.enabled_cases is None:return proposed
        mask=torch.zeros((base.shape[0],),device=base.device,dtype=torch.bool)
        for case in self.enabled_cases:
            if 0<=case<base.shape[0]:mask[case]=True
        return torch.where(mask[:,None],proposed,base)
