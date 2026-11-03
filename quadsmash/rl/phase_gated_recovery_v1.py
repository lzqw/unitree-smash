"""A phase-local residual head for recovery-only policy updates.

The inherited public actor remains the nominal full-action policy.  A separate
small head is added only when the current public observation reports the causal
post-flight foot-contact phase (index 774).  The nominal actor is frozen by the
training entry; the new head starts at zero, so initialization preserves the
source policy bit-for-bit before recovery learning begins.
"""
import math
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from .posthit_exploration_v1 import PostHitExplorationActorCriticV1


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite scalar')
    return float(value)


def _build_head(hidden_dims):
    if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
        raise ValueError('Declare at least one phase-head hidden layer')
    dims = [794]
    for value in hidden_dims:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Phase-head widths must be positive integers')
        dims.append(value)
    dims.append(18)
    modules = []
    for left, right in zip(dims[:-2], dims[1:-1]):
        modules.extend([nn.Linear(left, right), nn.ELU()])
    modules.append(nn.Linear(dims[-2], dims[-1]))
    head = nn.Sequential(*modules)
    final = head[-1]
    if not isinstance(final, nn.Linear):
        raise RuntimeError('Phase-head final layer must be Linear')
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return head


class _RecoveryExport(nn.Module):
    """Scriptable normalized public actor containing the phase-local head."""
    def __init__(self, normalizer, base, head, scale=1.0):
        super().__init__()
        self.normalizer = normalizer
        self.base = base
        self.head = head
        self.scale = float(scale)

    def forward(self, raw):
        normalized = self.normalizer(raw)
        gate = (raw[:, 774:775] > 0.5).to(raw.dtype)
        return self.base(normalized) + gate * self.head(normalized) * self.scale


class PhaseGatedRecoveryActorCriticV1(PostHitExplorationActorCriticV1):
    """Keep the learned pre-hit actor fixed and train a phase-774 residual head."""

    phase_gated_recovery = True

    def __init__(self, obs, obs_groups, num_actions, *,
                 recovery_head_hidden_dims=(128, 64), recovery_head_output_scale=1.0,
                 **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if num_actions != 18:
            raise ValueError('Require the original 18-action residual interface')
        if sum(obs[k].shape[-1] for k in obs_groups['policy']) != 794:
            raise ValueError('Require the original public794D policy interface')
        self.recovery_head = _build_head(recovery_head_hidden_dims)
        self.recovery_head_output_scale = _positive_scale(recovery_head_output_scale, 'recovery_head_output_scale')
        # Zero final weights make the transferred policy exactly unchanged at
        # initialization. Hidden weights remain finite and ready for PPO.
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def recovery_actor_parameters(self):
        """Parameters intentionally eligible for phase-local actor updates."""
        return list(self.recovery_head.parameters())

    def actor_update_parameters(self):
        """Compatibility hook used by retention/guidance optimizers."""
        return self.recovery_actor_parameters()

    def set_recovery_trainable(self, enabled):
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.recovery_head.parameters():
            parameter.requires_grad_(bool(enabled))

    def update_normalization(self, obs):
        """Keep the transferred public actor mapping fixed during recovery RL.

        The ordinary runner updates both normalizers during every rollout.  A
        frozen actor with a moving actor normalizer is still a moving policy,
        so only the critic statistics may adapt in this phase-local branch.
        """
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def _mean_for_raw(self, raw):
        normalized = self.actor_obs_normalizer(raw)
        gate = (raw[:, 774:775] > 0.5).to(raw.dtype)
        return self.actor(normalized) + gate * self.recovery_head(normalized) * self.recovery_head_output_scale

    def update_distribution(self, normalized_obs, *, raw_actor_obs=None):
        if raw_actor_obs is None:
            raise ValueError('Phase-gated recovery actor requires raw public phase observations')
        self.distribution = Normal(self._mean_for_raw(raw_actor_obs),
                                   self._std_for_raw_actor_obs(raw_actor_obs))

    def act(self, obs, **kwargs):
        raw = self.get_actor_obs(obs)
        self.update_distribution(self.actor_obs_normalizer(raw), raw_actor_obs=raw)
        return self.distribution.sample()

    def act_inference(self, obs):
        raw = self.get_actor_obs(obs)
        return self._mean_for_raw(raw)

    def export_deterministic_actor(self):
        """Return a CPU module with the exact raw public observation contract."""
        return _RecoveryExport(deepcopy(self.actor_obs_normalizer).cpu().eval(),
                               deepcopy(self.actor).cpu().eval(),
                               deepcopy(self.recovery_head).cpu().eval(),
                               self.recovery_head_output_scale)


def register_phase_gated_recovery_policy():
    """Register the class in rsl_rl's runner namespace before construction."""
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.PhaseGatedRecoveryActorCriticV1 = PhaseGatedRecoveryActorCriticV1


class _ApproachRecoveryExport(nn.Module):
    """Scriptable actor with separate pre-hit and post-hit residual heads."""
    def __init__(self, normalizer, base, approach, recovery, approach_scale, recovery_scale,
                 phase_start=771, phase_end=773):
        super().__init__()
        self.normalizer = normalizer
        self.base = base
        self.approach = approach
        self.recovery = recovery
        self.approach_scale = approach_scale
        self.recovery_scale = recovery_scale
        self.phase_start = phase_start
        self.phase_end = phase_end

    def forward(self, raw):
        normalized = self.normalizer(raw)
        approach_gate = (raw[:, self.phase_start:self.phase_end].sum(dim=1, keepdim=True) > 0.5).to(raw.dtype)
        recovery_gate = (raw[:, 774:775] > 0.5).to(raw.dtype)
        return (self.base(normalized)
                + approach_gate * self.approach(normalized) * self.approach_scale
                + recovery_gate * self.recovery(normalized) * self.recovery_scale)


class PhaseGatedApproachRecoveryActorCriticV1(PhaseGatedRecoveryActorCriticV1):
    """Frozen nominal actor with independent pre-hit and recovery heads."""

    def __init__(self, obs, obs_groups, num_actions, *,
                 approach_head_hidden_dims=(128, 64), approach_head_output_scale=1.0,
                 approach_phase_start_index=771, approach_phase_end_index=773, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        if (isinstance(approach_phase_start_index, bool)
                or not isinstance(approach_phase_start_index, int)
                or isinstance(approach_phase_end_index, bool)
                or not isinstance(approach_phase_end_index, int)
                or not 770 <= approach_phase_start_index < approach_phase_end_index <= 775):
            raise ValueError('Approach phase range must be within public phases 770:775')
        self.approach_head = _build_head(approach_head_hidden_dims)
        self.approach_head_output_scale = _positive_scale(
            approach_head_output_scale, 'approach_head_output_scale')
        self.approach_phase_start_index = approach_phase_start_index
        self.approach_phase_end_index = approach_phase_end_index

    def recovery_actor_parameters(self):
        return list(self.recovery_head.parameters()) + list(self.approach_head.parameters())

    def set_recovery_trainable(self, enabled):
        super().set_recovery_trainable(enabled)
        for parameter in self.approach_head.parameters():
            parameter.requires_grad_(bool(enabled))

    def _mean_for_raw(self, raw):
        normalized = self.actor_obs_normalizer(raw)
        approach_gate = (raw[:, self.approach_phase_start_index:self.approach_phase_end_index]
                         .sum(dim=1, keepdim=True) > 0.5).to(raw.dtype)
        recovery_gate = (raw[:, 774:775] > 0.5).to(raw.dtype)
        return (self.actor(normalized)
                + approach_gate * self.approach_head(normalized) * self.approach_head_output_scale
                + recovery_gate * self.recovery_head(normalized) * self.recovery_head_output_scale)

    def export_deterministic_actor(self):
        return _ApproachRecoveryExport(
            deepcopy(self.actor_obs_normalizer).cpu().eval(),
            deepcopy(self.actor).cpu().eval(),
            deepcopy(self.approach_head).cpu().eval(),
            deepcopy(self.recovery_head).cpu().eval(),
            self.approach_head_output_scale,
            self.recovery_head_output_scale,
            self.approach_phase_start_index,
            self.approach_phase_end_index)


def register_phase_gated_approach_recovery_policy():
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.PhaseGatedApproachRecoveryActorCriticV1 = PhaseGatedApproachRecoveryActorCriticV1

