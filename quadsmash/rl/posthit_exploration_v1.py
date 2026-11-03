"""Opt-in lower sampling variance during observed post-hit flight and recovery.

The phase is a current public observation. No time schedule, future outcome,
per-lane hidden latch, changed mean-action range or changed hardware limit is
introduced. Stock PPO receives the actual conditional Gaussian likelihood.
"""
import math

import torch

from .landing_exploration_v1 import (
    LandingExplorationActorCriticV1,
    exploration_contract as recovery_contract,
)


def positive_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Declare a positive finite post-hit exploration multiplier')
    return float(value)


def exploration_contract(policy_config):
    if policy_config['class_name'] not in ('PostHitExplorationActorCriticV1',
                                           'PhaseGatedRecoveryActorCriticV1',
                                           'PhaseGatedApproachRecoveryActorCriticV1',
                                           'FrozenV2ApproachActorCriticV1'):
        if 'airborne_hit_exploration_scale' in policy_config:
            raise ValueError('Select the post-hit actor explicitly instead of ignoring its multiplier')
        return recovery_contract(policy_config)
    scale = positive_scale(policy_config.get('airborne_hit_exploration_scale'))
    parent = dict(policy_config, class_name='LandingExplorationActorCriticV1')
    parent.pop('airborne_hit_exploration_scale')
    contract = recovery_contract(parent)
    contract.update(version='public-observed-hit-and-recovery-gaussian-v1',
        raw_public_airborne_hit_phase_index=773, airborne_hit_std_multiplier=scale,
        event_is_observed_contact_not_qualified_success=True, hidden_phase_state=False)
    if policy_config['class_name'] == 'FrozenV2ApproachActorCriticV1':
        contract.update(version='public-aerial-proximity-gaussian-v1',
            std_formula='posthit_conditional_std * current_public_proximity_gate + noise_floor',
            noise_floor_std=positive_scale(policy_config.get('approach_noise_floor_std',.0001)),
            proximity_core_distance_m=policy_config.get('approach_core_distance_m',.2),
            proximity_decay_length_m=policy_config.get('approach_decay_length_m',.9),
            observed_foot_history_frames=policy_config.get('approach_air_history_frames',6))
    return contract


class PostHitExplorationActorCriticV1(LandingExplorationActorCriticV1):
    def __init__(self, obs, obs_groups, num_actions, *, airborne_hit_exploration_scale, **kwargs):
        scale = positive_scale(airborne_hit_exploration_scale)
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.airborne_hit_exploration_scale = scale

    def _std_for_raw_actor_obs(self, raw):
        recovery_std = super()._std_for_raw_actor_obs(raw)
        base = self.log_std.exp() if self.noise_std_type == 'log' else self.std
        return torch.where(raw[:, 773:774] > .5,
                           base.expand(len(raw), -1) * self.airborne_hit_exploration_scale,
                           recovery_std)


def register_posthit_policy():
    """Register only inside the future training process; no installed-file edit."""
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.PostHitExplorationActorCriticV1 = PostHitExplorationActorCriticV1


def initialize_shared_target_actor(policy, configuration):
    from scripts.initialize_shared_target_actor_v6 import initialize_shared_target_actor as initialize_original
    result = initialize_original(policy, configuration)
    if isinstance(policy, PostHitExplorationActorCriticV1):
        std = torch.as_tensor(configuration['fresh_exploration_std'])
        scale = torch.as_tensor(configuration['joint_scale_rad'])
        result.update(
            physical_exploration_std_scope='Base sampling std outside observed post-hit airborne and recovery phases',
            airborne_hit_exploration_multiplier=policy.airborne_hit_exploration_scale,
            airborne_hit_physical_exploration_std_rad=(std * scale * policy.airborne_hit_exploration_scale).tolist())
    return result
