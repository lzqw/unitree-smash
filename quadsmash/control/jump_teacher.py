"""Bounded training-data controller; never used by policy acceptance evaluation."""
import torch

PARAMETER_NAMES = ('crouch_time','push_time','crouch_thigh','crouch_calf',
                   'extend_thigh','extend_calf','front_bias','tuck_time',
                   'tuck_thigh','tuck_calf','recovery_damping','recovery_posture',
                   'recovery_front_bias','pitch_feedback')
LOW = [.12,.06,.35,.35,.25,.35,-.25,.04, -.1,-1.,0.,-.25,-.3,-1.]
HIGH = [.38,.25,1.,1.,1.,1.,.35,.28, .6,-.1,.22,.3,.3,1.]

def teacher_action(observation, parameters):
    """Only current policy-observable clock, joint speed and gravity are read."""
    t = observation[:,775]
    ct,pt,cth,cca,eth,eca,bias,tt,tth,tca,damp,post,rbias,pgain = parameters.unbind(1)
    crouch=(t>=.8)&(t<.8+ct)
    push=(t>=.8+ct)&(t<.8+ct+pt)
    tuck=(t>=.8+ct+pt)&(t<.8+ct+pt+tt)
    recovery=t>=.8+ct+pt+tt
    action=torch.zeros((len(t),18),device=observation.device)
    action[:,4:8]=torch.where(crouch,cth,torch.where(push,-eth,torch.where(tuck,tth,post*recovery)))[:,None]
    action[:,8:12]=torch.where(crouch,-cca,torch.where(push,eca,torch.where(tuck,tca,-2*post*recovery)))[:,None]
    offset=push*bias+recovery*(rbias+pgain*observation[:,6])
    action[:,4:6]+=offset[:,None]; action[:,6:8]-=offset[:,None]
    # Active damping is an ordinary bounded position command, not a modified actuator.
    action[:,:12]-=(recovery*damp)[:,None]*observation[:,27:39]
    return action.clamp(-1,1)
