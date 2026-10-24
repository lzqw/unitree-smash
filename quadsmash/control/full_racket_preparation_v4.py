"""Public-reference task-space preparation; commands only, never state writes."""
import math
import torch


def validate_preparation(config):
    if config is None:return None
    point=config.get('position_world_m')
    if not isinstance(point,list) or len(point)!=3 or not all(math.isfinite(v) for v in point):
        raise ValueError('Finite3D preparation reference required')
    duration=float(config.get('transition_duration_s',.55))
    gain=float(config.get('servo_gain',1.))
    null_gain=float(config.get('nullspace_gain',.05))
    maximum=float(config.get('unhit_reference_max_x_m',1.4))
    if not .2<=duration<=1. or not 0.<gain<=1. or not 0.<=null_gain<=.2:
        raise ValueError('Invalid preparation duration/gain')
    if not point[0]<maximum<1.5 or not .5<=point[2]<=1.5:
        raise ValueError('Preparation reference must stay within declared own-side planning bounds')
    return dict(position_world_m=point,transition_duration_s=duration,servo_gain=gain,
                nullspace_gain=null_gain,unhit_reference_max_x_m=maximum)


def preparation_curve(time,event_time,preparation,target,velocity,duration):
    """C1 Hermite approach, followed by150ms smooth velocity decay.

    Positions and world-velocity vectors are expressed in the same current
    body frame. All references are constructed from public known commands.
    """
    u=((time-event_time+duration)/duration).clamp(0.,1.)[:,None]
    h00=2*u**3-3*u**2+1;h01=-2*u**3+3*u**2;h11=u**3-u**2
    position=h00*preparation+h01*target+h11*duration*velocity
    derivative=(6*u**2-6*u)/duration*preparation+(-6*u**2+6*u)/duration*target+(3*u**2-2*u)*velocity
    post=((time-event_time)/.15).clamp(0.,1.)[:,None]
    after=target+.15*(post-.5*post**2)*velocity
    after_velocity=(1-post)*velocity
    mask=(time>=event_time)[:,None]
    return torch.where(mask,after,position),torch.where(mask,after_velocity,derivative),u.squeeze(1)


def own_side_reference(position,velocity,known_command,forward,observed_hit,maximum_x):
    """Bound only the unhit commanded reference, not measured rule events."""
    x=((position-known_command)*forward).sum(1)+1.10
    excess=torch.where(observed_hit,torch.zeros_like(x),(x-maximum_x).clamp_min(0.))
    moving=((velocity*forward).sum(1)).clamp_min(0.)
    corrected=position-excess[:,None]*forward
    speed=velocity-torch.where(excess>0,moving,torch.zeros_like(moving))[:,None]*forward
    return corrected,speed
