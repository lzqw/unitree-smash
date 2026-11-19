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


def ball_transition(base,ball,state,targets,previous_action,feet,present,elapsed,launch,*,action_model=None):
    """The optional action model sees the CURRENT final command only through
    the command-conditioned robot prediction. Legacy mode is unchanged.
    """
    prior,_=base.prior(state,present,elapsed,launch)
    features=ball_geometry_features(state,base.geometry)
    mode_conditioned=getattr(action_model,'requires_racket_probability',False)
    if mode_conditioned:
        if not hasattr(ball,'components'):
            raise ValueError('Mode-conditioned sweep requires object-separated ball components')
        probability,response=ball.components(features)
        if probability.shape != (len(state),5) or response.shape != (len(state),4,6):
            raise ValueError('Unexpected object-separated ball mode schema')
        correction=present[:,None].to(response.dtype)*(probability[:,1:,None]*response).sum(1)
    else:
        correction,_=ball(features,present)
    physical,logits,active=base(state,targets,previous_action,feet,present,elapsed,launch)
    ball_prediction=prior[:,49:]+correction
    impulse_prior=getattr(action_model,'impulse_prior',None)
    if impulse_prior is not None:
        # The analytic prior is now part of the same state transition as the
        # learned residual. It receives the predicted current-command robot
        # state and therefore remains action-conditioned. Legacy models keep
        # the byte-compatible free-flight path unless they explicitly attach
        # this opt-in prior.
        analytic_ball, impulse_info = impulse_prior(
            state, base.geometry, present, predicted_robot=physical)
        ball_prediction=analytic_ball+correction
    else:
        impulse_info=None
    if action_model is not None:
        # V4 preserves its historical string-only contract. V5 explicitly
        # includes the learned frame-contact mode as a second causal impact
        # probability; native/future event labels never enter forward().
        if mode_conditioned:
            impact_probability=probability[:,1]
            if getattr(action_model,'requires_frame_probability',False):
                impact_probability=impact_probability+probability[:,4]
            kwargs={'racket_probability':impact_probability}
            if getattr(action_model,'requires_previous_action',False):
                kwargs['previous_action']=previous_action
            if getattr(action_model,'requires_current_target',False):
                kwargs['current_target']=targets
        else:
            kwargs={}
        response=action_model(state,physical,prior[:,49:],base.geometry,present,**kwargs)
        ball_prediction=ball_prediction+response
    state=torch.cat((physical[:,:49],ball_prediction),-1)
    return state,logits.sigmoid(),active
