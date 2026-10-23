"""Observable all-body strike controller with delayed arm preparation.

Schema: original 38 reach/swing/landing parameters, then arm raise start and
duration. It holds the selected walking arm posture until the raise begins.
"""
import torch
from .dynamic_reach_teacher import dynamic_reach_action


def composed_reach_action(obs,parameters,default,limits,arm,initial_arm):
    if parameters.shape[1]!=40:
        raise ValueError('composed-reach-v2 expects 40 parameters')
    action=dynamic_reach_action(obs,parameters[:,:38],default,limits,arm)
    t=obs['policy'][:,775]
    returning=((2.05-t)/.35).clamp(0,1)
    blend=((t-parameters[:,38])/parameters[:,39]).clamp(0,1)*returning
    q=torch.lerp(initial_arm[None,:],arm[None,:]+parameters[:,21:27],blend[:,None])
    swing=((t-parameters[:,33])/.06).clamp(0,1)*returning
    q+=parameters[:,27:33]*swing[:,None]
    action[:,12:]=(2*(q-limits[12:,0])/(limits[12:,1]-limits[12:,0])-1).clamp(-1,1)
    return action
