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


def ball_transition(base,ball,state,targets,previous_action,feet,present,elapsed,launch,*,
                    action_model=None,return_details=False,ball_input_grad=True):
    """The optional action model sees the CURRENT final command only through
    the command-conditioned robot prediction. Legacy mode is unchanged.
    """
    body_input = state
    if not ball_input_grad:
        state = state.detach()
    prior,_=base.prior(state,present,elapsed,launch)
    features=ball_geometry_features(state,base.geometry)
    impulse_prior=getattr(action_model,'impulse_prior',None)
    mode_conditioned=getattr(action_model,'requires_racket_probability',False)
    if impulse_prior is not None and not hasattr(ball,'components'):
        raise ValueError('Hybrid transition requires object-separated legacy ball components')
    if mode_conditioned or impulse_prior is not None:
        if not hasattr(ball,'components'):
            raise ValueError('Mode-conditioned sweep requires object-separated ball components')
        probability,response=ball.components(features)
        if probability.shape != (len(state),5) or response.shape != (len(state),4,6):
            raise ValueError('Unexpected object-separated ball mode schema')
        if impulse_prior is not None:
            # The analytic prior owns racket/ground/net contact.  Keep only
            # BallResidualV2's explicit ``other_residual`` mode so the old
            # contact heads cannot apply a second impulse at the same event.
            # ``response`` has four entries because the free mode is the gate
            # class, not a response head: racket, ground, net, other.
            correction=(present[:, None].to(response.dtype)
                        * probability[:, 4:5]
                        * response[:, 3])
        else:
            correction=present[:,None].to(response.dtype)*(probability[:,1:,None]*response).sum(1)
    else:
        correction,_=ball(features,present)
    physical,logits,active=base(body_input,targets,previous_action,feet,present,elapsed,launch)
    robot_prediction = physical[:,:49]
    if not ball_input_grad:
        physical = physical.detach()
        targets = targets.detach()
        previous_action = previous_action.detach()
    ball_prediction=prior[:,49:]+correction
    launch_now=(~present.bool()) & (elapsed.reshape(-1) >=
                                    getattr(base,'launch_s',2.) - 1e-6)
    if impulse_prior is not None:
        # The analytic prior is now part of the same state transition as the
        # learned residual. It receives the predicted current-command robot
        # state and therefore remains action-conditioned. Legacy models keep
        # the byte-compatible free-flight path unless they explicitly attach
        # this opt-in prior.
        defer_post_impact = bool(getattr(
            action_model, 'requires_impulse_residual', False)
            and getattr(impulse_prior, 'supports_deferred_post_impact', False))
        prior_kwargs = dict(predicted_robot=physical)
        if defer_post_impact:
            prior_kwargs['defer_post_impact'] = True
        analytic_ball, impulse_info = impulse_prior(
            state, base.geometry, present, **prior_kwargs)
        # Launch is an external schedule transition.  It has priority over
        # the collision prior, which intentionally leaves parked-ball states
        # unchanged until the ball is present.
        ball_prediction=torch.where(launch_now[:,None], prior[:,49:],
                                    analytic_ball+correction)
    else:
        impulse_info=None
    action_residual=torch.zeros_like(ball_prediction)
    impulse_action_residual=None
    action_model_info=None
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
        need_action_details = return_details or getattr(
            action_model, 'requires_impulse_residual', False)
        if need_action_details:
            kwargs['details']=True
        if getattr(action_model, 'requires_impulse_residual', False):
            if impulse_info is None:
                raise ValueError('TOI impulse residual requires an attached analytic prior')
            kwargs['impact_info'] = impulse_info
        # The sweep head predicts pre-impact support.  Keep its endpoint on
        # the free-flight side of the transition.  Feeding the already
        # impacted analytic state back into this gate would let the response
        # change the event it is meant to model.
        action_ball_endpoint = prior[:,49:]
        response=action_model(state,physical,action_ball_endpoint,
                              base.geometry,present,**kwargs)
        if need_action_details:
            response,action_model_info=response
        action_residual=response
        ball_prediction=ball_prediction+action_residual
        if action_model_info is not None and 'impulse_residual_dv' in action_model_info:
            if impulse_prior is None or impulse_info is None:
                raise ValueError('Impulse residual metadata requires analytic prior')
            residual_dv=action_model_info['impulse_residual_dv']
            # Re-integrate from the same continuous TOI state used by the
            # analytic prior.  Adding a raw six-dimensional end-state offset
            # would mix impulse units with position units and would apply the
            # correction at the wrong time.
            corrected_position, corrected_velocity = impulse_prior.advance(
                impulse_info['pre_position'],
                impulse_info['pre_velocity'] + impulse_info['impulse_dv'] + residual_dv,
                impulse_info['remaining_time'])
            corrected_ball=torch.cat((corrected_position, corrected_velocity),-1)
            analytic_ball=action_model_info.get('analytic_ball', analytic_ball)
            if analytic_ball is None:
                raise ValueError('Analytic ball state is required for TOI residual')
            impulse_mask=(
                impulse_info['active']
                & (impulse_info.get('object_id', torch.zeros_like(
                    impulse_info['active'], dtype=torch.long)) == 0)
                & present.reshape(-1).bool())
            impulse_action_residual=corrected_ball-analytic_ball
            impulse_action_residual=impulse_action_residual * impulse_mask[:,None].to(
                impulse_action_residual.dtype)
            deferred_mask=impulse_info.get('deferred_post_impact')
            if deferred_mask is not None:
                # In deferred mode analytic_ball contains the common TOI
                # pre-state plus the analytic impulse velocity.  Apply the
                # learned impulse and integrate once from that state.  Rows
                # without a deferred racket impact retain the normal analytic
                # transition and legacy residual path.
                corrected_prediction=corrected_ball+correction+action_residual
                ball_prediction=torch.where(
                    deferred_mask[:,None], corrected_prediction, ball_prediction)
            else:
                ball_prediction=ball_prediction+impulse_action_residual
    state=torch.cat((robot_prediction,ball_prediction),-1)
    next_feet=logits.sigmoid()
    if return_details:
        return state,next_feet,active,dict(
            predicted_robot=robot_prediction,
            prior_ball=prior[:,49:],
            analytic_ball=None if impulse_info is None else analytic_ball,
            legacy_correction=correction,
            action_residual=action_residual,
            impulse_action_residual=impulse_action_residual,
            predicted_ball=ball_prediction,
            impulse_info=impulse_info,
            action_model_info=action_model_info,
            launch_now=launch_now,
            hybrid=impulse_prior is not None,
        )
    return state,next_feet,active
