"""Reset-safe event windows and full-state autoregressive ball supervision."""
import torch
from .supermimic_physical_dataset_v1 import valid_windows
from .supermimic_ball_residual_v1 import ball_geometry_features


def ball_window_pools(data,masks,lanes,steps=16):
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],steps)
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[lanes]=True
    valid &= allowed[None]
    pools={'uniform':valid.nonzero()}
    for name in ('new_racket','new_ground','new_net'):
        event=masks[name].unfold(0,steps,1).any(-1)&valid
        if event.any():pools[name]=event.nonzero()
    return pools


def ball_error(predicted,target):
    # Physical units, fixed before the run. These are training scales only.
    scale=predicted.new_tensor([.1,.1,.1,2.,2.,2.])
    return ((predicted-target)/scale).square().mean()


def ball_transition(base,ball,state,targets,previous_action,feet,present,elapsed,launch):
    prior,_=base.prior(state,present,elapsed,launch)
    correction,_=ball(ball_geometry_features(state,base.geometry),present)
    physical,logits,active=base(state,targets,previous_action,feet,present,elapsed,launch)
    state=torch.cat((physical[:,:49],prior[:,49:]+correction),-1)
    return state,logits.sigmoid(),active
