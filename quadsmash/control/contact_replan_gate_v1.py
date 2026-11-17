"""Per-lane bounded failure retry gate; no candidate or outcome case lookup."""
import torch


class ContactReplanGate:
    def __init__(self,count,device,interval_s=.16,max_attempts=4,close_s=2.64):
        self.interval_s=interval_s;self.max_attempts=max_attempts;self.close_s=close_s
        self.count=torch.zeros(count,dtype=torch.long,device=device)
        self.last=torch.full((count,),-float('inf'),device=device)
        self.stopped=torch.zeros(count,dtype=torch.bool,device=device)
    def reset(self,ids=None):
        ids=slice(None) if ids is None else ids
        self.count[ids]=0;self.last[ids]=-float('inf');self.stopped[ids]=False
    def eligible(self,public,verified_plan,recovery_owned):
        # Contact/post-hit phase or Recovery permanently closes this episode.
        self.stopped |= recovery_owned | (public[:,773]>.5)
        clock=public[:,775]
        return ((public[:,67]>.5)&(clock>=2.)&(clock<=self.close_s)&
                (self.count<self.max_attempts)&(clock-self.last>=self.interval_s-1e-6)&
                ~self.stopped&~verified_plan)
    def record(self,ids,clock):
        self.count[ids]+=1;self.last[ids]=clock[ids]
