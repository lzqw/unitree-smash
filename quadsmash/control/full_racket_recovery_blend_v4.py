"""Bounded leg-target handover using the observed previous action."""
import torch


class LegRecoveryBlend:
    def __init__(self,count,device,seconds=0.):
        if not 0.<=seconds<=.25:
            raise ValueError('Recovery blend must be between0 and.25seconds')
        self.seconds=float(seconds)
        self.started=torch.zeros(count,dtype=torch.bool,device=device)
        self.start_time=torch.zeros(count,device=device)
        self.start_leg=torch.zeros((count,12),device=device)

    def reset(self,ids=None):
        if ids is None:
            self.started.zero_();self.start_time.zero_();self.start_leg.zero_()
        else:
            self.started[ids]=False;self.start_time[ids]=0.;self.start_leg[ids]=0.

    def action(self,time,active,observed_previous_action,standing_action):
        if self.seconds==0.:
            return standing_action
        self.started[~active]=False
        beginning=active&(~self.started)
        self.start_time[beginning]=time[beginning]
        self.start_leg[beginning]=observed_previous_action[beginning,:12]
        self.started|=active
        fraction=((time-self.start_time)/self.seconds).clamp(0.,1.)
        fraction=fraction.square()*(3-2*fraction)
        result=standing_action.clone()
        result[:,:12]=torch.lerp(self.start_leg,standing_action[:,:12],fraction[:,None]).clamp(-1.,1.)
        return result
