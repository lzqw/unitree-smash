"""Frozen learned Go2 actor for measured post-strike landing and standing.

Only public proprioception enters the 48-dimensional actor. Each lane begins
with fresh action history and holds the actor output for two control steps.
The five searched parameters change bounded leg targets, never hardware.
"""
import torch

from .go2_walk_transfer import Go2WalkTransfer


class Go2StandingTransfer(Go2WalkTransfer):
    def __init__(self, manifest, num_envs, device):
        super().__init__(manifest, num_envs, device)
        self.standing_ticks = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self):
        super().reset()
        self.standing_ticks.zero_()

    def action(self, observation, parameters, default, limits, arm_target, active=None):
        obs = observation['policy']
        if active is None:
            active = torch.ones(len(obs), dtype=torch.bool, device=obs.device)
        update = active & (self.standing_ticks.remainder(2) == 0)
        source = torch.cat((obs[:, :9], torch.zeros_like(obs[:, :3]),
                            obs[:, 9:21], obs[:, 27:39], self.last_raw), 1)
        raw = self.actor(source)
        self.last_raw.copy_(torch.where(update[:, None], raw, self.last_raw))
        self.standing_ticks.add_(active.long())
        q = default[None, :].repeat(len(obs), 1)
        q[:, :12] += .25*self.last_raw*parameters[:, 4:5]
        q[:, 4:8] += parameters[:, 0:1]
        q[:, 4:6] += parameters[:, 1:2]
        q[:, 6:8] -= parameters[:, 1:2]
        q[:, 8:12] += parameters[:, 2:3]
        q[:, :4] += parameters[:, 3:4]*obs[:, 7:8]
        q[:, 12:] = arm_target
        return (2*(q-limits[:, 0])/(limits[:, 1]-limits[:, 0])-1).clamp(-1, 1)
