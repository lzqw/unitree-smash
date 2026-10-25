"""TRAIN rewards from measured events; never substitutes acceptance labels."""
import torch

EVENT_WEIGHTS=dict(flight=3.,air=40.,attack=15.,crossed=10.,target=20.,
                  contact_recovery=20.,complete=150.,robot_fault=-20.,body_fault=-8.)


class ResidualEventRewardV4:
    def __init__(self,count,device):
        self.paid={name:torch.zeros(count,dtype=torch.bool,device=device) for name in EVENT_WEIGHTS}

    def reset(self,ids=None):
        for paid in self.paid.values():
            if ids is None:paid.zero_()
            else:paid[ids]=False

    def step(self,flags,dense,motion_cost,dt):
        if dt<=0 or not torch.isfinite(dense).all() or not torch.isfinite(motion_cost).all():
            raise ValueError('Finite integrated TRAIN reward terms required')
        reward=(3*dense-motion_cost)*dt
        for name,weight in EVENT_WEIGHTS.items():
            flag=flags[name]
            if flag.shape!=self.paid[name].shape or flag.dtype!=torch.bool:
                raise ValueError('Require one measured boolean event per lane')
            first=flag&(~self.paid[name]);self.paid[name]|=flag
            reward=reward+weight*first.float()
        return reward
