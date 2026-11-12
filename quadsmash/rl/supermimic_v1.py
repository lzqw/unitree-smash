"""A compact SuperMimic-style model-based learner for the full-racket task.

This module deliberately does not import PPO, old checkpoints, teachers, MPC, or
demonstration data.  It uses native transitions to fit a latent-free one-step
world model, then performs differentiable imagined rollouts through that model.
The implementation is intentionally explicit so every transition can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.LayerNorm(width), nn.SiLU()]
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SuperMimicWorldModel(nn.Module):
    """Predict observation delta, reward and terminal probability."""
    def __init__(self, obs_dim: int, action_dim: int = 18):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.body = MLP(obs_dim + action_dim, (1024, 1024, 1024), obs_dim + 2)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        out = self.body(torch.cat((obs, action), dim=-1))
        delta = out[..., : self.obs_dim]
        reward = out[..., self.obs_dim : self.obs_dim + 1]
        terminal_logit = out[..., self.obs_dim + 1 :]
        return delta, reward, terminal_logit


class SuperMimicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 18):
        super().__init__()
        self.mean = MLP(obs_dim, (512, 256, 128), action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.4))

    def forward(self, obs: torch.Tensor, sample: bool = False):
        mean = self.mean(obs)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = dist.rsample() if sample else mean
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        log_prob -= torch.log1p(-action.square() + 1e-6).sum(-1, keepdim=True)
        return action, log_prob, dist.entropy().sum(-1, keepdim=True)


class SuperMimicCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.value = MLP(obs_dim, (512, 256, 128), 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs)


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class EventReplay:
    """Device-resident replay with ordinary/current-rare/history-rare sampling.

    Isaac observations already live on the learner device.  Keeping the ring
    buffer on that same device removes the high-frequency simulator→CPU→GPU
    copy that otherwise dominates small native rollouts.  Checkpointing can
    optionally materialize a CPU copy at a low-frequency save boundary.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity, self.device = capacity, device
        self.storage_device = device
        self.obs = torch.empty((capacity, obs_dim), device=device)
        self.action = torch.empty((capacity, action_dim), device=device)
        self.next_obs = torch.empty((capacity, obs_dim), device=device)
        self.reward = torch.empty((capacity, 1), device=device)
        self.done = torch.empty((capacity, 1), device=device)
        self.rare = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.ptr = 0
        self.size = 0
        self.rare_history = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(self, obs, action, next_obs, reward, done, rare):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        obs_dev, action_dev = obs.detach(), action.detach()
        next_dev = next_obs.detach()
        reward_dev = reward.detach().reshape(-1, 1)
        done_dev = done.detach().reshape(-1, 1)
        rare_dev = rare.detach().reshape(-1).to(dtype=torch.bool)
        overwritten = self.rare[idx]
        self.rare_history[idx] = overwritten
        self.obs[idx], self.action[idx], self.next_obs[idx] = obs_dev, action_dev, next_dev
        self.reward[idx], self.done[idx], self.rare[idx] = reward_dev, done_dev.float(), rare_dev
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < batch_size:
            raise RuntimeError(f"replay has {self.size} transitions, needs {batch_size}")
        active = torch.arange(self.size, device=self.device)
        current = active[self.rare[: self.size]]
        historical = active[self.rare_history[: self.size]]
        ordinary = active[~self.rare[: self.size] & ~self.rare_history[: self.size]]
        counts = [batch_size // 2, batch_size // 4, batch_size - (batch_size // 2 + batch_size // 4)]
        pools = [ordinary, current, historical]
        chosen = []
        for pool, count in zip(pools, counts):
            if pool.numel() == 0:
                pool = active
            chosen.append(pool[torch.randint(pool.numel(), (count,), device=self.device)])
        idx = torch.cat(chosen)
        idx = idx[torch.randperm(idx.numel(), device=self.device)]
        return ReplayBatch(self.obs[idx], self.action[idx], self.next_obs[idx],
                           self.reward[idx], self.done[idx])

    def state_dict(self, include_data: bool = False) -> dict:
        state = {'capacity': self.capacity, 'ptr': self.ptr, 'size': self.size}
        if include_data:
            # Materialize only at an explicit low-frequency checkpoint boundary.
            state.update(obs=self.obs[:self.size].cpu(), action=self.action[:self.size].cpu(),
                         next_obs=self.next_obs[:self.size].cpu(), reward=self.reward[:self.size].cpu(),
                         done=self.done[:self.size].cpu(), rare=self.rare[:self.size].cpu(),
                         rare_history=self.rare_history[:self.size].cpu())
        return state

    def load_state_dict(self, state: dict) -> None:
        if not state or 'obs' not in state:
            return
        n = min(int(state.get('size', state['obs'].shape[0])), self.capacity)
        self.obs[:n].copy_(state['obs'][:n].to(self.device))
        self.action[:n].copy_(state['action'][:n].to(self.device))
        self.next_obs[:n].copy_(state['next_obs'][:n].to(self.device))
        self.reward[:n].copy_(state['reward'][:n].to(self.device))
        self.done[:n].copy_(state['done'][:n].to(self.device))
        self.rare[:n].copy_(state['rare'][:n].to(self.device))
        self.rare_history[:n].copy_(state['rare_history'][:n].to(self.device))
        self.size, self.ptr = n, int(state.get('ptr', n % self.capacity))


class SuperMimicTrainer:
    """Native collection + world-model fitting + imagined actor/critic updates."""
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, *, lr=2e-4,
                 gamma=0.998, imagine_horizon=32, replay_capacity=250_000,
                 model_warmup_steps=1000):
        self.device, self.gamma, self.horizon = device, gamma, imagine_horizon
        self.actor = SuperMimicActor(obs_dim, action_dim).to(device)
        self.critic = SuperMimicCritic(obs_dim).to(device)
        self.world = SuperMimicWorldModel(obs_dim, action_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=lr)
        self.replay = EventReplay(replay_capacity, obs_dim, action_dim, device)
        self.model_warmup_steps = model_warmup_steps
        self.update_steps = 0

    @torch.no_grad()
    def action(self, obs: torch.Tensor, sample: bool = True):
        return self.actor(obs, sample=sample)[0].clamp(-1.0, 1.0)

    def fit_world(self, updates: int, batch_size: int) -> dict:
        if len(self.replay) < batch_size:
            return {"world_updates": 0, "world_loss": None}
        losses = []
        for _ in range(updates):
            b = self.replay.sample(batch_size)
            delta, reward, terminal = self.world(b.obs, b.action)
            loss = ((delta - (b.next_obs - b.obs)).square().mean()
                    + 0.5 * (reward - b.reward).square().mean()
                    + 0.1 * (terminal - b.done).square().mean())
            self.world_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.world.parameters(), 10.0)
            self.world_opt.step()
            losses.append(float(loss.detach()))
        return {"world_updates": updates, "world_loss": sum(losses) / len(losses)}

    def imagine_update(self, batch_size: int, critic_weight: float = 0.5, entropy_weight: float = 1e-3) -> dict:
        if len(self.replay) < batch_size:
            return {"imagine_update": False}
        start = self.replay.sample(batch_size).obs.detach()
        # Freeze world-model parameters for the actor update while retaining
        # its input Jacobian, i.e. gradients flow through predictions but the
        # model is not updated by imagined loss.
        old_flags = [p.requires_grad for p in self.world.parameters()]
        for p in self.world.parameters(): p.requires_grad_(False)
        imagined = start
        rewards, values, entropies = [], [], []
        for _ in range(self.horizon):
            action, _, entropy = self.actor(imagined, sample=False)
            delta, reward, terminal = self.world(imagined, action)
            values.append(self.critic(imagined))
            rewards.append(reward)
            entropies.append(entropy)
            imagined = (imagined + delta).clamp(-20.0, 20.0)
            # The model terminal is a soft stop; retaining the state keeps the
            # graph stable while its probability is reflected in the return.
            imagined = imagined * (1.0 - terminal.sigmoid().detach())
        returns = torch.zeros_like(rewards[-1])
        actor_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        for reward, value in reversed(list(zip(rewards, values))):
            returns = reward + self.gamma * returns
            actor_loss = actor_loss - (returns.detach() - value.detach()).mean() * 0.0 - returns.mean()
            critic_loss = critic_loss + (value - returns.detach()).square().mean()
        actor_loss = actor_loss / self.horizon - entropy_weight * torch.cat(entropies, -1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for p, flag in zip(self.world.parameters(), old_flags): p.requires_grad_(flag)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss = critic_weight * critic_loss / self.horizon
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()
        self.update_steps += 1
        return {"imagine_update": True, "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()), "imagined_return": float(returns.mean().detach())}

    def save(self, path: Path, metadata: dict, *, include_replay: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": "supermimic_v1", "random_initialization": True,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "world_model": self.world.state_dict(), "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(), "world_opt": self.world_opt.state_dict(),
                    "update_steps": self.update_steps, "replay": self.replay.state_dict(include_replay),
                    "metadata": metadata | {'replay_persisted': bool(include_replay)},
                    }, path)

    def load(self, path: Path, *, restore_replay: bool = True) -> dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('algorithm') != 'supermimic_v1' or state.get('metadata', {}).get('old_checkpoint') is not None:
            raise ValueError('resume checkpoint is not an independent SuperMimic checkpoint')
        self.actor.load_state_dict(state['actor']); self.critic.load_state_dict(state['critic'])
        self.world.load_state_dict(state['world_model'])
        if state.get('actor_opt'): self.actor_opt.load_state_dict(state['actor_opt'])
        if state.get('critic_opt'): self.critic_opt.load_state_dict(state['critic_opt'])
        if state.get('world_opt'): self.world_opt.load_state_dict(state['world_opt'])
        self.update_steps = int(state.get('update_steps', 0))
        if restore_replay and state.get('replay', {}).get('obs') is not None:
            self.replay.load_state_dict(state['replay'])
        return state

    def load_actor(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("algorithm") != "supermimic_v1":
            raise ValueError("checkpoint is not a SuperMimic checkpoint")
        self.actor.load_state_dict(state["actor"])
        return state
