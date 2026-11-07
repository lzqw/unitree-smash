"""Causal post-hit support latch, independent of any learned recovery head.

Matches the existing student's eligibility: previously observed airborne hit,
then current support phase; stay active through transient support loss until
the real episode clock resets. No case identity or outcome is an input.
"""
import torch


class ObservedPostHitGateV1:
    def __init__(self, count, device, dt=.01):
        self.count=count; self.device=device; self.dt=dt
        self.known=torch.zeros(count,dtype=torch.bool,device=device)
        self.hit=torch.zeros_like(self.known); self.active=torch.zeros_like(self.known)
        self.clock=torch.zeros(count,device=device)

    @torch.inference_mode()
    def update(self, public):
        if public.shape!=(self.count,794) or not torch.isfinite(public).all():
            raise ValueError('Finite original public observations required')
        clock=public[:,775]
        reset=self.known & (clock<self.clock-1e-6)
        self.known[reset]=False; self.hit[reset]=False; self.active[reset]=False
        if torch.any((~self.known)&(abs(clock)>1e-6)):
            raise ValueError('Must begin from a real ground reset')
        if torch.any(self.known & (abs(clock-self.clock-self.dt)>2e-6)):
            raise ValueError('Exactly one gate update per native control tick required')
        self.active |= self.known & self.hit & (public[:,774]>.5)
        self.hit |= public[:,773]>.5
        self.known[:]=True; self.clock.copy_(clock)
        return self.active.clone()
