"""Opt-in recovery-phase Gaussian exploration with stock PPO likelihoods.

The raw public recovery phase selects a variance multiplier. The deterministic
actor, action authority, hardware limits and original success criteria remain
the same. This is not a landing controller or a learned performance claim.
"""
import math

import torch
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


def exploration_contract(policy_config):
    name = policy_config['class_name']
    if name == 'ActorCritic':
        if 'landing_exploration_scale' in policy_config:
            raise ValueError('Select the conditional actor explicitly instead of ignoring its scale')
        return None
    if name != 'LandingExplorationActorCriticV1':
        raise ValueError('Unsupported public feedforward actor class')
    scale = policy_config.get('landing_exploration_scale')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Declare a positive finite recovery exploration scale')
    return dict(version='public-recovery-phase-gaussian-v1', raw_public_phase_index=774,
        recovery_std_multiplier=float(scale), deterministic_actor_unchanged=True,
        likelihood_uses_conditional_std=True)


class LandingExplorationActorCriticV1(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, *, landing_exploration_scale, **kwargs):
        scale = landing_exploration_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Declare a positive finite recovery exploration scale')
        if num_actions != 18 or sum(obs[k].shape[-1] for k in obs_groups['policy']) != 794:
            raise ValueError('Require the original public794D/18-action interface')
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        # An explicit training configuration field, included in the runner's
        # checkpoint signature. No extra dynamic state or phase latch is used.
        self.landing_exploration_scale = float(scale)

    def _std_for_raw_actor_obs(self, raw):
        if raw.ndim != 2 or raw.shape[1] != 794:
            raise ValueError('Require raw public observations before normalization')
        base = self.log_std.exp() if self.noise_std_type == 'log' else self.std
        # Public phase4 means observed recovery after an airborne interval.
        # It does not assert a legal hit or a task-qualified jump, and can also
        # occur after an earlier short flight during preparation.
        scale = torch.where(raw[:, 774:775] > .5, self.landing_exploration_scale, 1.)
        return base.expand(len(raw), -1) * scale

    def action_std_for_observations(self, obs):
        """RNG-free conditional standard deviation for PPO diagnostics."""
        return self._std_for_raw_actor_obs(self.get_actor_obs(obs))

    def update_distribution(self, normalized_obs, *, raw_actor_obs=None):
        if raw_actor_obs is None:
            raise ValueError('Conditional exploration requires raw phase observations; use act()')
        self.distribution = Normal(self.actor(normalized_obs), self._std_for_raw_actor_obs(raw_actor_obs))

    def act(self, obs, **kwargs):
        raw = self.get_actor_obs(obs)
        self.update_distribution(self.actor_obs_normalizer(raw), raw_actor_obs=raw)
        return self.distribution.sample()
