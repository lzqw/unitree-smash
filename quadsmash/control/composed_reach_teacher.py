"""Observable all-body strike controller with delayed arm preparation.

Schema: original 38 reach/swing/landing parameters, then arm raise start and
duration. Optional v3 parameter 40 compensates measured body pitch at the
shoulder. It holds the selected walking arm posture until the raise begins.
"""
import torch
from .dynamic_reach_teacher import dynamic_reach_action


def composed_reach_action(obs,parameters,default,limits,arm,initial_arm):
    if parameters.shape[1] not in (40,41,44):
        raise ValueError('composed reach expects v2/40, v3/41 or v4/44 parameters')
    action=dynamic_reach_action(obs,parameters[:,:38],default,limits,arm)
    t=obs['policy'][:,775]
    returning=(((parameters[:,41]+parameters[:,42]-t)/parameters[:,42]).clamp(0,1)
               if parameters.shape[1]==44 else ((2.05-t)/.35).clamp(0,1))
    blend=((t-parameters[:,38])/parameters[:,39]).clamp(0,1)*returning
    q=torch.lerp(initial_arm[None,:],arm[None,:]+parameters[:,21:27],blend[:,None])
    if parameters.shape[1]>=41:
        gravity=obs['policy'][:,6:9]
        pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        q[:,1]-=parameters[:,40]*pitch*blend
    swing=((t-parameters[:,33])/.06).clamp(0,1)*returning
    q+=parameters[:,27:33]*swing[:,None]
    if parameters.shape[1]==44:
        q[:,1]-=parameters[:,43]*(1-returning)
    action[:,12:]=(2*(q-limits[12:,0])/(limits[12:,1]-limits[12:,0])-1).clamp(-1,1)
    return action
