"""A compact SuperMimic-style model-based learner for the full-racket task.

The learner can start from a strictly checked native PPO actor, but does not
use PPO updates, teachers, MPC, or demonstration data during SuperMimic
updates. It uses native transitions to fit a latent-free one-step world model,
then performs differentiable imagined rollouts through that model.
The implementation is intentionally explicit so every transition can be audited.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import io
from pathlib import Path
from typing import Optional
import hashlib

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
    """Predict task transitions while preserving the native history contract.

    The native 794D observation contains eleven 69D frames.  Treating all
    759 history values as independently predicted physical state lets the
    imagination model invent inconsistent histories.  For the full-racket
    contract, this model predicts only the next frame and the non-history
    tail; the history shift itself is deterministic and identical to
    ``PolicyHistory.observe``.  The public forward interface remains
    ``(delta, reward, terminal_logit)`` so the existing learner can use the
    structured transition without an algorithm rewrite.
    """
    def __init__(self, obs_dim: int, action_dim: int = 18, *, structured_history: bool = True,
                 event_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.event_dim = int(event_dim)
        if self.event_dim < 0:
            raise ValueError("event_dim must be non-negative")
        self.structured_history = bool(structured_history and obs_dim == 794)
        self.history_length = 11
        self.frame_dim = 69
        self.history_dim = self.history_length * self.frame_dim
        self.tail_dim = obs_dim - self.history_dim
        output_dim = (self.frame_dim + self.tail_dim + 2 + self.event_dim
                      if self.structured_history else obs_dim + 2 + self.event_dim)
        self.body = MLP(obs_dim + action_dim + self.event_dim, (1024, 1024, 1024), output_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, event_state=None):
        if self.event_dim:
            if event_state is None or event_state.shape[-1] != self.event_dim:
                raise ValueError("event_state is required for an event-aware world model")
            model_input = torch.cat((obs, action, event_state), dim=-1)
        else:
            model_input = torch.cat((obs, action), dim=-1)
        out = self.body(model_input)
        if self.structured_history:
            frame_delta = out[..., :self.frame_dim]
            tail_delta = out[..., self.frame_dim:self.frame_dim + self.tail_dim]
            history = obs[..., :self.history_dim].reshape(
                *obs.shape[:-1], self.history_length, self.frame_dim)
            next_frame = history[..., 0, :] + frame_delta
            next_history = torch.cat((next_frame.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            next_tail = obs[..., self.history_dim:] + tail_delta
            next_obs = torch.cat((next_history.reshape(*obs.shape[:-1], self.history_dim), next_tail), dim=-1)
            delta = next_obs - obs
            reward = out[..., self.frame_dim + self.tail_dim:self.frame_dim + self.tail_dim + 1]
            terminal_logit = out[..., self.frame_dim + self.tail_dim + 1:self.frame_dim + self.tail_dim + 2]
            next_event_logits = out[..., self.frame_dim + self.tail_dim + 2:]
        else:
            delta = out[..., : self.obs_dim]
            reward = out[..., self.obs_dim : self.obs_dim + 1]
            terminal_logit = out[..., self.obs_dim + 1:self.obs_dim + 2]
            next_event_logits = out[..., self.obs_dim + 2:]
        if self.event_dim:
            return delta, reward, terminal_logit, next_event_logits
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


class PPOInitializedActor(nn.Module):
    """Exact actor contract used by the native 794D PPO policy.

    This deliberately keeps the old MLP and observation statistics instead of
    silently loading them into the unrelated LayerNorm/SiLU actor above.
    """
    actor_kind = "native_ppo_794_512_256_128_elu"
    def __init__(self, obs_dim: int, action_dim: int = 18):
        super().__init__()
        if obs_dim != 794 or action_dim != 18:
            raise ValueError(f"PPO actor contract requires 794/18, got {obs_dim}/{action_dim}")
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.0))

    def forward(self, obs: torch.Tensor, sample: bool = False):
        normalized = (obs - self.obs_mean) / self.obs_std.clamp_min(1e-6)
        mean = self.actor(normalized)
        std = self.log_std.clamp(-8.0, 1.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = dist.rsample() if sample else mean
        # Native PPO emits an unconstrained Gaussian action and the
        # environment clamps it at the action boundary.  Keeping this exact
        # contract is required for zero-update compatibility with jump-event-v1.
        action = raw
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        return action, log_prob, dist.entropy().sum(-1, keepdim=True)


class ResidualPPOInitializedActor(nn.Module):
    """Trainable wrapper around the measured full-racket residual actor.

    The native residual checkpoint is a TorchScript actor with a frozen base
    controller and a normalized 18D residual request.  The environment applies
    the original base+mixer execution contract before PhysX.  Keeping the
    residual request as the learner action lets replay/world-model training use
    the same public action interface while preserving the physical executor.
    """
    actor_kind = "native_residual_jit_794_256_128_64"

    def __init__(self, actor_module: nn.Module, action_dim: int = 18):
        super().__init__()
        self.actor = actor_module
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.0))
        self.early_arm_gate = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # This submodule is optional so legacy checkpoints retain their exact layout.
        if any(k.startswith('release_arm_reference.') for k in state_dict):
            self.install_release_arm_reference()
        elif getattr(self, 'release_arm_reference', None) is not None:
            del self.release_arm_reference
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def install_release_arm_reference(self):
        if getattr(self, 'release_arm_reference', None) is None:
            self.release_arm_reference = deepcopy(self.actor.recovery)
            for parameter in self.release_arm_reference.parameters():
                parameter.requires_grad_(False)

    def configure_approach_gate(self, preparation_seconds):
        from .frozen_v2_approach_v1 import EarlyArmApproachGate
        self.early_arm_gate = None if preparation_seconds is None else EarlyArmApproachGate(preparation_seconds)

    def forward(self, obs: torch.Tensor, sample: bool = False):
        if getattr(self, 'release_arm_reference', None) is not None:
            raise ValueError('Delayed Recovery requires causal release time and the explicit gate')
        mean = self.actor(obs)
        std = self.log_std.clamp(-8.0, 1.0).exp().expand_as(mean)
        dist = Normal(mean, std)
        raw = dist.rsample() if sample else mean
        action = raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        return action, log_prob, dist.entropy().sum(-1, keepdim=True)

    def forward_with_recovery_gate(self, obs: torch.Tensor, recovery_gate: torch.Tensor,
                                   sample: bool = False, release_time=None):
        """Evaluate the full-racket actor with an explicit causal recovery latch.

        The verified exported actor historically gates its recovery head from
        public phase4 (foot contact).  Phase4 also occurs after a missed ball,
        so it is not a sufficient post-hit signal.  This path preserves the
        base and approach computation and replaces only the recovery-head gate
        with the caller's per-lane ``had_contact & released`` latch.  The
        ordinary ``forward`` path remains byte-compatible for legacy callers.
        """
        dist = self._gated_distribution(obs, recovery_gate, release_time=release_time)
        raw = dist.rsample() if sample else dist.loc
        action = raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        return action, log_prob, dist.entropy().sum(-1, keepdim=True)

    def gated_request(self, obs: torch.Tensor, recovery_gate: torch.Tensor):
        """Deterministic imagined control uses the validated mean of the native request."""
        return self._gated_distribution(obs, recovery_gate).loc.clamp(-1., 1.)

    def _gated_distribution(self, obs: torch.Tensor, recovery_gate: torch.Tensor, release_time=None):
        if recovery_gate.ndim == 1:
            recovery_gate = recovery_gate[:, None]
        if recovery_gate.shape != (obs.shape[0], 1):
            raise ValueError('recovery_gate must have shape [batch,1]')
        module = self.actor
        required = ('normalizer', 'base', 'recovery', 'approach', 'gate',
                    'recovery_scale', 'approach_scale')
        if not all(hasattr(module, name) for name in required):
            raise ValueError('Explicit recovery gate requires FrozenV2 approach export')
        normalized = module.normalizer(obs)
        recovery = module.recovery(normalized)
        reference = getattr(self, 'release_arm_reference', None)
        if reference is not None:
            if release_time is None or release_time.numel() != len(obs):
                raise ValueError('Recovery arm delay requires native per-lane release times')
            ready = obs[:, 775:776] - release_time.reshape(-1, 1) >= .04
            recovery = torch.cat((recovery[:, :12], torch.where(
                ready, recovery[:, 12:], reference(normalized)[:, 12:])), dim=-1)
        mean = (module.base(normalized)
                + recovery_gate.to(obs.dtype) * recovery * module.recovery_scale
                + self.approach_action_gate(obs) * module.approach(normalized) * module.approach_scale)
        std = self.log_std.clamp(-8.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def approach_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-joint gate for commands; legacy checkpoints use one broadcast value."""
        module = self.actor
        if not hasattr(module, 'gate'):
            raise ValueError('Approach gate requires FrozenV2 approach export')
        legacy = module.gate(obs)
        return legacy if self.early_arm_gate is None else self.early_arm_gate(obs, legacy)

    def approach_gate(self, obs: torch.Tensor) -> torch.Tensor:
        """Lane activity used by replay and imagination selection."""
        return self.approach_action_gate(obs).amax(dim=-1, keepdim=True)

    def recovery_gate(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the legacy exported recovery gate for diagnostics only."""
        module = self.actor
        if not hasattr(module, 'gate'):
            raise ValueError('Recovery gate requires FrozenV2 approach export')
        return module.gate(obs)


def load_ppo_initialized_actor(path: Path, device: torch.device):
    """Load the native PPO actor with strict, auditable interface checks."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get('schema') in ('full-racket-residual-ppo-v4', 'g07-shared-target-residual-ppo-v1'):
        blob = state.get('residual_actor_jit')
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError('Residual PPO policy has no embedded TorchScript actor')
        actor_module = torch.jit.load(io.BytesIO(blob), map_location=device)
        actor = ResidualPPOInitializedActor(actor_module).to(device)
        actor.eval()
        interface = dict(schema='quadsmash-native-training-v2', observation_size=794,
                         history_length=11, joints=state.get('base_controller', {}).get('seed', {}).get('joints', []),
                         policy_dt=0.01,
                         action_family='bounded-residual-joint-targets-full-racket-v4')
        if not interface['joints']:
            interface['joints'] = [
                'FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint',
                'FL_thigh_joint','FR_thigh_joint','RL_thigh_joint','RR_thigh_joint',
                'FL_calf_joint','FR_calf_joint','RL_calf_joint','RR_calf_joint',
                'joint1','joint2','joint3','joint4','joint5','joint6']
        return actor, {
            'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'native_schema': state.get('schema'), 'training_signature': state.get('residual_training_metadata', {}).get('training_signature'),
            'interface': interface, 'learning_rate': state.get('residual_training_metadata', {}).get('learning_rate'),
            'actor_kind': actor.actor_kind, 'obs_dim': 794, 'action_dim': 18,
            'execution_policy': ('frozen_g07_shared_target_base_plus_native_residual_mixer'
                                 if state.get('schema') == 'g07-shared-target-residual-ppo-v1'
                                 else 'frozen_full_racket_base_plus_native_residual_mixer'),
            'g07_execution_root': '/home/autodrive/LZQW/unitree_smash'
            if state.get('schema') == 'g07-shared-target-residual-ppo-v1' else None,
        }
    model = state.get("model_state_dict")
    native = state.get("native", {})
    if not isinstance(model, dict):
        raise ValueError("PPO checkpoint has no model_state_dict")
    if int(native.get("observation_size", 794)) != 794:
        raise ValueError("PPO observation contract is not 794D")
    joint_order = native.get("joint_order", native.get("interface", {}).get("joints", []))
    if joint_order and len(joint_order) != 18:
        raise ValueError("PPO joint order is not 18 joints")
    required = ["actor.0.weight", "actor.2.weight", "actor.4.weight", "actor.6.weight",
                "actor_obs_normalizer._mean", "actor_obs_normalizer._std", "log_std"]
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"PPO checkpoint missing exact actor keys: {missing}")
    actor = PPOInitializedActor(794, 18)
    mapped = {
        "actor.0.weight": model["actor.0.weight"], "actor.0.bias": model["actor.0.bias"],
        "actor.2.weight": model["actor.2.weight"], "actor.2.bias": model["actor.2.bias"],
        "actor.4.weight": model["actor.4.weight"], "actor.4.bias": model["actor.4.bias"],
        "actor.6.weight": model["actor.6.weight"], "actor.6.bias": model["actor.6.bias"],
        "obs_mean": model["actor_obs_normalizer._mean"].reshape(-1),
        "obs_std": model["actor_obs_normalizer._std"].reshape(-1),
        "log_std": model["log_std"],
    }
    # The mapping is explicit and strict; no partial/strict=False loading.
    actor_state = {"actor." + k.split(".", 1)[1]: v for k, v in mapped.items() if k.startswith("actor.")}
    actor_state.update({"obs_mean": mapped["obs_mean"], "obs_std": mapped["obs_std"], "log_std": mapped["log_std"]})
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device)
    actor.eval()
    raw_interface = native.get("interface", {}) if isinstance(native, dict) else {}
    interface = {key: raw_interface.get(key) for key in
                  ("schema", "observation_size", "history_length", "joints",
                  "policy_dt", "action_family") if key in raw_interface}
    return actor, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_schema": native.get("schema"),
        "training_signature": native.get("training_signature"),
        "interface": interface,
        "learning_rate": native.get("learning_rate"),
        "actor_kind": actor.actor_kind,
        "obs_dim": 794, "action_dim": 18,
    }


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
        self.stream = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.episode = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.prev = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.next = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.last_by_stream: dict[int, int] = {}
        archive_capacity = max(1024, capacity // 4)
        self.archive_capacity = archive_capacity
        self.archive_obs = torch.empty((archive_capacity, obs_dim), device=device)
        self.archive_action = torch.empty((archive_capacity, action_dim), device=device)
        self.archive_next_obs = torch.empty((archive_capacity, obs_dim), device=device)
        self.archive_reward = torch.empty((archive_capacity, 1), device=device)
        self.archive_done = torch.empty((archive_capacity, 1), device=device)
        self.archive_rare = torch.zeros(archive_capacity, dtype=torch.bool, device=device)
        self.archive_ptr = 0
        self.ptr = 0
        self.size = 0
        self.rare_history = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(self, obs, action, next_obs, reward, done, rare, *, stream=None, episode=None):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        obs_dev, action_dev = obs.detach(), action.detach()
        next_dev = next_obs.detach()
        reward_dev = reward.detach().reshape(-1, 1)
        done_dev = done.detach().reshape(-1, 1)
        rare_dev = rare.detach().reshape(-1).to(dtype=torch.bool)
        overwritten = self.rare[idx]
        # Preserve overwritten rare transitions in a separate archive instead
        # of retaining a boolean attached to the newly written slot.
        rare_idx = torch.where(overwritten)[0].detach().cpu().tolist()
        for j in rare_idx:
            dst = self.archive_ptr % self.archive_capacity
            src = int(idx[j].item())
            self.archive_obs[dst].copy_(self.obs[src])
            self.archive_action[dst].copy_(self.action[src])
            self.archive_next_obs[dst].copy_(self.next_obs[src])
            self.archive_reward[dst].copy_(self.reward[src])
            self.archive_done[dst].copy_(self.done[src])
            self.archive_rare[dst] = True
            self.archive_ptr += 1
        self.rare_history[idx] = False
        self.obs[idx], self.action[idx], self.next_obs[idx] = obs_dev, action_dev, next_dev
        self.reward[idx], self.done[idx], self.rare[idx] = reward_dev, done_dev.float(), rare_dev
        if stream is None:
            stream = torch.arange(n, device=self.device)
        if episode is None:
            episode = torch.zeros(n, dtype=torch.long, device=self.device)
        stream = stream.detach().to(self.device, dtype=torch.long).reshape(-1)
        episode = episode.detach().to(self.device, dtype=torch.long).reshape(-1)
        self.stream[idx] = stream
        self.episode[idx] = episode
        self.prev[idx] = -1
        self.next[idx] = -1
        for j in range(n):
            slot = int(idx[j].item()); sid = int(stream[j].item())
            previous = self.last_by_stream.get(sid, -1)
            if previous >= 0 and bool(self.episode[previous].item() == int(episode[j].item())):
                self.prev[slot] = previous
                self.next[previous] = slot
            self.last_by_stream[sid] = slot
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
        chosen = []
        for pool, count in zip((ordinary, current), counts[:2]):
            if pool.numel() == 0:
                pool = active
            chosen.append(pool[torch.randint(pool.numel(), (count,), device=self.device)])
        historical_count = counts[2]
        archive_size = min(self.archive_ptr, self.archive_capacity)
        if archive_size:
            hist_idx = torch.randint(archive_size, (historical_count,), device=self.device)
            hist = ReplayBatch(self.archive_obs[hist_idx], self.archive_action[hist_idx],
                               self.archive_next_obs[hist_idx], self.archive_reward[hist_idx],
                               self.archive_done[hist_idx])
            main_idx = torch.cat(chosen)
            main = ReplayBatch(self.obs[main_idx], self.action[main_idx], self.next_obs[main_idx],
                               self.reward[main_idx], self.done[main_idx])
            return ReplayBatch(torch.cat((main.obs, hist.obs)), torch.cat((main.action, hist.action)),
                               torch.cat((main.next_obs, hist.next_obs)), torch.cat((main.reward, hist.reward)),
                               torch.cat((main.done, hist.done)))
        chosen.append(active[torch.randint(active.numel(), (historical_count,), device=self.device)])
        idx = torch.cat(chosen)
        idx = idx[torch.randperm(idx.numel(), device=self.device)]
        return ReplayBatch(self.obs[idx], self.action[idx], self.next_obs[idx],
                           self.reward[idx], self.done[idx])

    def sample_windows(self, batch_size: int, horizon: int) -> list[ReplayBatch]:
        """Sample contiguous replay windows, rejecting reset boundaries."""
        horizon = max(1, int(horizon))
        if self.size < batch_size + horizon - 1:
            return [self.sample(batch_size)]
        starts = torch.randint(0, self.size, (batch_size * 4,), device=self.device)
        idx = torch.full((starts.numel(), horizon), -1, dtype=torch.long, device=self.device)
        idx[:, 0] = starts
        for k in range(1, horizon):
            prior = idx[:, k - 1]
            idx[:, k] = torch.where(prior >= 0, self.next[prior.clamp_min(0)], torch.full_like(prior, -1))
        valid = (idx >= 0).all(-1)
        valid &= (self.stream[idx.clamp_min(0)] == self.stream[idx[:, :1].clamp_min(0)]).all(-1)
        valid &= (self.episode[idx.clamp_min(0)] == self.episode[idx[:, :1].clamp_min(0)]).all(-1)
        valid &= (self.done[idx[:, :-1].clamp_min(0), 0] < 0.5).all(-1)
        valid_idx = idx[valid][:batch_size]
        if valid_idx.shape[0] < max(1, batch_size // 2):
            return [self.sample(batch_size)]
        return [ReplayBatch(self.obs[valid_idx[:, k]], self.action[valid_idx[:, k]],
                            self.next_obs[valid_idx[:, k]], self.reward[valid_idx[:, k]],
                            self.done[valid_idx[:, k]]) for k in range(horizon)]

    def state_dict(self, include_data: bool = False) -> dict:
        state = {'capacity': self.capacity, 'ptr': self.ptr, 'size': self.size}
        if include_data:
            # Materialize only at an explicit low-frequency checkpoint boundary.
            state.update(obs=self.obs[:self.size].cpu(), action=self.action[:self.size].cpu(),
                         next_obs=self.next_obs[:self.size].cpu(), reward=self.reward[:self.size].cpu(),
                         done=self.done[:self.size].cpu(), rare=self.rare[:self.size].cpu(),
                         rare_history=self.rare_history[:self.size].cpu(),
                         stream=self.stream[:self.size].cpu(), episode=self.episode[:self.size].cpu(),
                         prev=self.prev[:self.size].cpu(), next=self.next[:self.size].cpu())
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
        if 'stream' in state:
            self.stream[:n].copy_(state['stream'][:n].to(self.device))
            self.episode[:n].copy_(state['episode'][:n].to(self.device))
            self.prev[:n].copy_(state['prev'][:n].to(self.device))
            self.next[:n].copy_(state['next'][:n].to(self.device))
        self.size, self.ptr = n, int(state.get('ptr', n % self.capacity))


class SuperMimicTrainer:
    """Native collection + world-model fitting + imagined actor/critic updates."""
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, *, lr=2e-4,
                 gamma=0.998, imagine_horizon=32, replay_capacity=250_000,
                 model_warmup_steps=1000, actor: Optional[nn.Module] = None,
                 actor_kind: str = "supermimic_random"):
        self.device, self.gamma, self.horizon = device, gamma, imagine_horizon
        self.actor = (actor if actor is not None else SuperMimicActor(obs_dim, action_dim)).to(device)
        self.actor_kind = actor_kind
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
            windows = self.replay.sample_windows(batch_size, min(4, self.horizon))
            loss = torch.zeros((), device=self.device)
            survival = torch.ones((windows[0].obs.shape[0], 1), device=self.device)
            predicted_obs = windows[0].obs
            for b in windows:
                delta, reward, terminal = self.world(predicted_obs, b.action)
                # The world model is trained autoregressively from its own
                # predicted state; use the recorded next state only as target.
                target_delta = b.next_obs - predicted_obs.detach()
                wm_loss = (delta - target_delta).square().mean(-1, keepdim=True)
                wm_loss = wm_loss + 0.5 * (reward - b.reward).square()
                wm_loss = wm_loss + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
                    terminal, b.done, reduction='none')
                loss = loss + survival * wm_loss
                predicted_obs = predicted_obs + delta
                survival = survival * (1.0 - b.done).detach()
            loss = loss.mean() / len(windows)
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
        critic_flags = [p.requires_grad for p in self.critic.parameters()]
        for p in self.world.parameters(): p.requires_grad_(False)
        for p in self.critic.parameters(): p.requires_grad_(False)
        imagined = start
        rewards, states, survival, entropies = [], [], [], []
        for _ in range(self.horizon):
            action, _, entropy = self.actor(imagined, sample=False)
            action = action.clamp(-1.0, 1.0)
            delta, reward, terminal = self.world(imagined, action)
            states.append(imagined)
            rewards.append(reward)
            entropies.append(entropy)
            alive = (1.0 - terminal.sigmoid()).clamp(0.0, 1.0)
            survival.append(alive)
            imagined = (imagined + delta).clamp(-20.0, 20.0)
        # Keep critic parameters frozen but retain dV/ds for actor BPTT.
        bootstrap = self.critic(imagined)
        returns = bootstrap
        targets = []
        for reward, alive in reversed(list(zip(rewards, survival))):
            returns = reward + self.gamma * alive * returns
            targets.append(returns)
        targets.reverse()
        actor_loss = -torch.stack(targets).mean() - entropy_weight * torch.cat(entropies, -1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for p, flag in zip(self.world.parameters(), old_flags): p.requires_grad_(flag)
        for p, flag in zip(self.critic.parameters(), critic_flags): p.requires_grad_(flag)
        # Re-forward the detached imagination for critic-only optimization;
        # actor/world graphs never leak into this update.
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss = torch.zeros((), device=self.device)
        for state, target in zip(states, targets):
            value = self.critic(state.detach())
            critic_loss = critic_loss + (value - target.detach()).square().mean()
        critic_loss = critic_weight * critic_loss / self.horizon
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()
        self.update_steps += 1
        return {"imagine_update": True, "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()), "imagined_return": float(targets[0].mean().detach())}

    def save(self, path: Path, metadata: dict, *, include_replay: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": "supermimic_v1", "random_initialization": self.actor_kind == "supermimic_random",
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "world_model": self.world.state_dict(), "actor_opt": self.actor_opt.state_dict(),
                    "critic_opt": self.critic_opt.state_dict(), "world_opt": self.world_opt.state_dict(),
                    "update_steps": self.update_steps, "replay": self.replay.state_dict(include_replay),
                    "metadata": metadata | {'replay_persisted': bool(include_replay), 'actor_kind': self.actor_kind},
                    }, path)

    def load(self, path: Path, *, restore_replay: bool = True) -> dict:
        # Deserialize on CPU first; mapping the large optimizer/world-model
        # checkpoint directly into a live Isaac CUDA context can block Kit
        # startup. State tensors are moved explicitly below.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get('algorithm') != 'supermimic_v1' or state.get('metadata', {}).get('old_checkpoint') is not None:
            raise ValueError('resume checkpoint is not an independent SuperMimic checkpoint')
        expected_kind = state.get('metadata', {}).get('actor_kind', 'supermimic_random')
        if expected_kind != self.actor_kind:
            raise ValueError(f"resume actor kind {expected_kind} != current {self.actor_kind}")
        self.actor.load_state_dict(state['actor']); self.critic.load_state_dict(state['critic'])
        self.world.load_state_dict(state['world_model'])
        if state.get('actor_opt'): self.actor_opt.load_state_dict(state['actor_opt'])
        if state.get('critic_opt'): self.critic_opt.load_state_dict(state['critic_opt'])
        if state.get('world_opt'): self.world_opt.load_state_dict(state['world_opt'])
        for optimizer in (self.actor_opt, self.critic_opt, self.world_opt):
            for slot in optimizer.state.values():
                for key, value in list(slot.items()):
                    if torch.is_tensor(value): slot[key] = value.to(self.device)
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
