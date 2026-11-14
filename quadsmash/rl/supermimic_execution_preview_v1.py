"""Pure, differentiable preview of the ORIGINAL residual mixer and arm limiter.

Requires the actual base-controller output. Does not pretend to reproduce the
nested G07 base controller's planning memory. Not enabled in native execution.
"""
from dataclasses import dataclass

import torch

from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4


@dataclass(frozen=True)
class MixerMemory:
    arm_target: torch.Tensor
    arm_clock: torch.Tensor
    arm_initialized: torch.Tensor
    nonzero_seen: torch.Tensor

    @classmethod
    def capture(cls, mixer):
        return cls(mixer.arm_rate.target.detach().clone(),
                   mixer.arm_rate.clock.detach().clone(),
                   mixer.arm_rate.initialized.clone(), mixer.nonzero_seen.clone())

    def reset(self, ids):
        # Native reset only clears flags; target/clock remain but are ignored
        # at the next call because initialized=False.
        initialized, nonzero = self.arm_initialized.clone(), self.nonzero_seen.clone()
        initialized[ids] = False
        nonzero[ids] = False
        return MixerMemory(self.arm_target.clone(), self.arm_clock.clone(), initialized, nonzero)


@dataclass(frozen=True)
class MixerPreview:
    requested_normalized: torch.Tensor
    final_normalized: torch.Tensor
    final_joint_targets: torch.Tensor
    next_memory: MixerMemory


class ResidualExecutionPreviewV1(torch.nn.Module):
    def __init__(self, joint_scale_rad, arm_rate_rad_s=5., max_scale_rad=.25, control_dt_s=.01):
        super().__init__()
        # Delegate config validation to the real implementation, not a second
        # looser limits definition. This object is never used for execution.
        original = ResidualActionMixerV4(1, 'cpu', joint_scale_rad,
                                        arm_rate_rad_s, max_scale_rad)
        if float(control_dt_s) != original.arm_rate.control_dt:
            raise ValueError('Preview must preserve the original mixer control period')
        self.register_buffer('scale', original.scale.clone())
        self.rate = original.arm_rate.rate
        self.control_dt = original.arm_rate.control_dt

    def forward(self, base, residual, public, default, limits, memory: MixerMemory):
        if base.shape != residual.shape or base.shape != (len(memory.nonzero_seen), 18):
            raise ValueError('Require one18D base and residual command per lane')
        if not torch.isfinite(base).all() or not torch.isfinite(residual).all():
            raise ValueError('Finite residual/base commands required')
        residual = residual.clamp(-1., 1.)
        nonzero = memory.nonzero_seen | (residual != 0).any(1)
        requested = (base+2*residual*self.scale/(limits[:, 1]-limits[:, 0])).clamp(-1., 1.)
        desired = limits[:, 0]+.5*(requested+1)*(limits[:, 1]-limits[:, 0])
        observed = (default[12:][None]+public[:, 21:27]).clamp(limits[12:, 0], limits[12:, 1])
        time = public[:, 775]
        if not all(torch.isfinite(x).all() for x in (desired, observed, time)):
            raise ValueError('Finite public inputs required for arm target rate control')
        restarted = (~memory.arm_initialized) | (time < memory.arm_clock-1e-6)
        before = torch.where(restarted[:, None], observed, memory.arm_target)
        dt = torch.where(restarted, torch.full_like(time, self.control_dt),
                         (time-memory.arm_clock).clamp(0., self.control_dt))
        maximum_step = self.rate*dt[:, None]
        bounded = before+(desired[:, 12:]-before).clamp(-maximum_step, maximum_step)
        arm = (2*(bounded-limits[12:, 0])/(limits[12:, 1]-limits[12:, 0])-1).clamp(-1., 1.)
        final = torch.where(nonzero[:, None], torch.cat((requested[:, :12], arm), -1), base)
        targets = limits[:, 0]+.5*(final.clamp(-1., 1.)+1)*(limits[:, 1]-limits[:, 0])
        next_memory = MixerMemory(bounded, time.clone(), torch.ones_like(restarted), nonzero)
        return MixerPreview(requested, final, targets, next_memory)
