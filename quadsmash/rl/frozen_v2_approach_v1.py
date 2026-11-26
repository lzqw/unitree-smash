"""Learn an aerial correction over the entire frozen, verified v2 policy.

The old nominal actor, its normalizer and its phase774 recovery head form one
immutable policy. Only a new head is optimized. Its continuous gate uses
measured foot-contact history and ball/tool separation; changing the hit flag
does not switch the correction off. No future outcomes or hidden latches enter
the deployed actor or PPO likelihood.
"""
from copy import deepcopy
from typing import Optional

import torch
from torch import nn

from .phase_gated_recovery_v1 import (
    PhaseGatedApproachRecoveryActorCriticV1, _positive_scale,
)


class PublicAerialProximityGate(nn.Module):
    def __init__(self, core_distance_m=0.2, decay_length_m=0.9, history_frames=6):
        super().__init__()
        self.core_distance_m = _positive_scale(core_distance_m, 'core_distance_m')
        self.decay_length_m = _positive_scale(decay_length_m, 'decay_length_m')
        if isinstance(history_frames, bool) or not isinstance(history_frames, int) or not 1 <= history_frames <= 11:
            raise ValueError('Use 1..11 actual public history frames')
        self.history_frames = history_frames

    def forward(self, raw):
        # Public frames are newest first, each of width 69. Indices are fixed
        # by observation_contract.py: feet57:61, tool45:48, ball61:64,
        # ball-present67 and history-valid68.
        history = raw[:, :759].reshape(-1, 11, 69)[:, :self.history_frames]
        valid = history[:, :, 68].clamp(0., 1.)
        supported = history[:, :, 57:61].amax(dim=2).clamp(0., 1.)
        aerial = ((1. - supported) * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.)
        distance = (raw[:, 61:64] - raw[:, 45:48]).norm(p=2, dim=1, keepdim=True)
        separation = ((distance - self.core_distance_m) / self.decay_length_m).clamp_min(0.)
        proximity = torch.exp(-separation.square())
        return aerial * proximity * raw[:, 67:68].clamp(0., 1.)


class EarlyArmApproachGate(nn.Module):
    """Causal swing preview from the current public ball and racket frame."""
    def __init__(self, preparation_seconds=.25):
        super().__init__()
        self.preparation_seconds = _positive_scale(preparation_seconds, 'preparation_seconds')

    def forward(self, raw, legacy):
        displacement = raw[:, 61:64] - raw[:, 45:48]
        relative = raw[:, 64:67] - raw[:, 51:54]
        closing = -(displacement * relative).sum(-1, keepdim=True)
        closest_time = closing / relative.square().sum(-1, keepdim=True).clamp_min(1e-6)
        closest = displacement + closest_time.clamp(0., self.preparation_seconds) * relative
        # Fade into the preparation interval, retaining the old aerial gate
        # through contact. A receding shuttle cannot start a new early swing.
        ramp = ((self.preparation_seconds - closest_time) / .05).clamp(0., 1.)
        ramp = ramp.square() * (3. - 2. * ramp)
        proximity = torch.exp(-(closest.norm(p=2, dim=-1, keepdim=True) / .6).square())
        post_hit = (raw[:, 773:774] > .5) | (raw[:, 774:775] > .5)
        early = ramp * proximity * (closing > 0.) * raw[:, 67:68].clamp(0., 1.) * ~post_hit
        return torch.cat((legacy.expand(-1, 12), torch.maximum(legacy, early).expand(-1, 6)), -1)


class CausalApproachActorExport(nn.Module):
    """Deterministic residual request with the two native Recovery latches as inputs."""

    def __init__(self, actor):
        super().__init__()
        self.policy = deepcopy(actor.actor).eval()
        self.early_arm_gate = deepcopy(actor.early_arm_gate)
        self.release_reference = deepcopy(getattr(actor, "release_arm_reference", None))

    @torch.jit.export
    def action_gate(self, observation: torch.Tensor) -> torch.Tensor:
        legacy = self.policy.gate(observation)
        if self.early_arm_gate is None:
            return legacy.expand(-1, 18)
        return self.early_arm_gate(observation, legacy)

    def forward(self, observation: torch.Tensor, had_contact: torch.Tensor,
                released: torch.Tensor, release_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        torch._assert(observation.ndim == 2 and observation.size(1) == 794, 'Expected [batch,794] observation')
        torch._assert(had_contact.numel() == observation.size(0)
                      and released.numel() == observation.size(0), 'Expected one causal latch per lane')
        normalized = self.policy.normalizer(observation)
        recovery = (had_contact.reshape(-1, 1) > .5) & (released.reshape(-1, 1) > .5)
        head = self.policy.recovery(normalized)
        if self.release_reference is not None:
            if release_time is None or release_time.numel() != observation.size(0):
                raise ValueError('Recovery arm delay requires one release time per lane')
            ready = observation[:, 775:776] - release_time.reshape(-1, 1) >= .04
            head = torch.cat((head[:, :12], torch.where(
                ready, head[:, 12:], self.release_reference(normalized)[:, 12:])), dim=-1)
        raw = (self.policy.base(normalized)
               + recovery.to(observation.dtype) * head * self.policy.recovery_scale
               + self.action_gate(observation) * self.policy.approach(normalized) * self.policy.approach_scale)
        return raw.clamp(-1., 1.)


class _FrozenV2ApproachExport(nn.Module):
    def __init__(self, normalizer, base, recovery, approach, gate, recovery_scale, approach_scale):
        super().__init__()
        self.normalizer, self.base = normalizer, base
        self.recovery, self.approach = recovery, approach
        self.gate = gate
        self.recovery_scale, self.approach_scale = recovery_scale, approach_scale

    def forward(self, raw):
        normalized = self.normalizer(raw)
        old_recovery = (raw[:, 774:775] > .5).to(raw.dtype)
        return (self.base(normalized)
                + old_recovery * self.recovery(normalized) * self.recovery_scale
                + self.gate(raw) * self.approach(normalized) * self.approach_scale)


class FrozenV2ApproachActorCriticV1(PhaseGatedApproachRecoveryActorCriticV1):
    freeze_verified_policy = True

    def __init__(self, obs, obs_groups, num_actions, *,
                 approach_core_distance_m=0.2, approach_decay_length_m=0.9,
                 approach_air_history_frames=6, approach_noise_floor_std=0.0001, **kwargs):
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.approach_gate = PublicAerialProximityGate(
            approach_core_distance_m, approach_decay_length_m, approach_air_history_frames)
        self.approach_noise_floor_std = _positive_scale(approach_noise_floor_std,'approach_noise_floor_std')
        self.set_recovery_trainable(False)
        self._frozen_reference = None

    def recovery_actor_parameters(self):
        # Existing rehearsal optimizers call actor_update_parameters(), which
        # delegates here. They must never acquire the old recovery parameters.
        return list(self.approach_head.parameters())

    def set_recovery_trainable(self, enabled):
        for module in (self.actor, self.actor_obs_normalizer, self.recovery_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.log_std.requires_grad_(False)
        for parameter in self.approach_head.parameters():
            parameter.requires_grad_(bool(enabled))

    def _mean_for_raw(self, raw):
        normalized = self.actor_obs_normalizer(raw)
        gate = (raw[:, 774:775] > .5).to(raw.dtype)
        frozen = self.actor(normalized) + gate * self.recovery_head(normalized) * self.recovery_head_output_scale
        return frozen + self.approach_gate(raw) * self.approach_head(normalized) * self.approach_head_output_scale

    def _std_for_raw_actor_obs(self, raw):
        # Explore where the new module can act. The tiny positive background
        # variance keeps a valid Gaussian likelihood without shaking the
        # otherwise immutable standing/landing controller each control tick.
        return (super()._std_for_raw_actor_obs(raw) * self.approach_gate(raw)
                + self.approach_noise_floor_std)

    def _frozen_state(self):
        return {k: v for k, v in self.state_dict().items()
                if k == 'log_std' or k.startswith(('actor.', 'actor_obs_normalizer.', 'recovery_head.'))}

    def capture_frozen_reference(self):
        self._frozen_reference = {k: v.detach().clone() for k, v in self._frozen_state().items()}
        return self.assert_frozen_reference()

    def assert_frozen_reference(self):
        if self._frozen_reference is None:
            raise RuntimeError('Capture the transferred full-policy state before learning')
        current = self._frozen_state()
        changed = [k for k, v in self._frozen_reference.items() if not torch.equal(current[k], v)]
        if changed:
            raise RuntimeError('Verified frozen policy changed: ' + ', '.join(changed))
        return dict(frozen_tensors=len(current), frozen_actor_and_normalizer=True,
                    frozen_recovery_head=True, frozen_log_std=True, all_bitwise_unchanged=True,
                    trainable_module='approach_head')

    def retention_sampling_weights(self, raw):
        return self.approach_gate(raw).squeeze(-1)

    def export_deterministic_actor(self):
        return _FrozenV2ApproachExport(
            deepcopy(self.actor_obs_normalizer).cpu().eval(),
            deepcopy(self.actor).cpu().eval(),
            deepcopy(self.recovery_head).cpu().eval(),
            deepcopy(self.approach_head).cpu().eval(), deepcopy(self.approach_gate).cpu().eval(),
            self.recovery_head_output_scale, self.approach_head_output_scale)


class FrozenApproachRecoveryRepairActorCriticV1(FrozenV2ApproachActorCriticV1):
    """Freeze a measured approach policy and train only its recovery head."""

    phase_gated_recovery = True
    freeze_verified_policy = True

    def recovery_actor_parameters(self):
        return list(self.recovery_head.parameters())

    def set_recovery_trainable(self, enabled):
        for module in (self.actor, self.actor_obs_normalizer, self.approach_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.log_std.requires_grad_(False)
        for parameter in self.recovery_head.parameters():
            parameter.requires_grad_(bool(enabled))

    def _std_for_raw_actor_obs(self, raw):
        recovery = (raw[:, 774:775] > .5).to(raw.dtype)
        base = self.log_std.exp() if self.noise_std_type == 'log' else self.std
        return base.expand(len(raw), -1) * recovery + .0001

    def _frozen_state(self):
        return {k: v for k, v in self.state_dict().items()
                if k == 'log_std' or k.startswith(('actor.', 'actor_obs_normalizer.', 'approach_head.'))}

    def assert_frozen_reference(self):
        if self._frozen_reference is None:
            raise RuntimeError('Capture the transferred approach policy before learning')
        current = self._frozen_state()
        changed = [k for k, v in self._frozen_reference.items() if not torch.equal(current[k], v)]
        if changed:
            raise RuntimeError('Verified frozen approach policy changed: ' + ', '.join(changed))
        return dict(frozen_tensors=len(current), frozen_actor_and_normalizer=True,
                    frozen_approach_head=True, frozen_log_std=True,
                    all_bitwise_unchanged=True, trainable_module='recovery_head')

    def retention_sampling_weights(self, raw):
        return (raw[:, 774:775] > .5).to(raw.dtype).squeeze(-1)


def register_frozen_v2_approach_policy():
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.FrozenV2ApproachActorCriticV1 = FrozenV2ApproachActorCriticV1
    runner_module.FrozenApproachRecoveryRepairActorCriticV1 = FrozenApproachRecoveryRepairActorCriticV1
