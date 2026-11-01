"""Optional soft command prior on current public phases 0/1/2.

This changes training reward only. All 18 residual actions still pass through
the existing mixer, limits and actuators. The prior is not a teacher-success
label, action mask, trajectory tracker or new observation.
"""
import math

import torch


SCHEMA = 'public-phase-prehit-residual-command-cost-v1'


def settings(value):
    if not isinstance(value, dict) or set(value) != {'weight_per_s', 'reference_scale_rad'}:
        raise ValueError('Declare exactly weight_per_s and reference_scale_rad')
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError('Prior settings must be finite numbers')
        if item < 0 or (key == 'reference_scale_rad' and item == 0):
            raise ValueError('Require nonnegative weight and positive reference scale')
        result[key] = float(item)
    return result


def command_cost(public, residual, joint_scale_rad, dt, configuration):
    """Use the observation preceding the action, including terminal transitions.

The physical-unit cost applies to requested clipped residuals, before joint
range and slew limiting. Phase 0 can occur after an earlier contact as well;
we intentionally use only the current public phase, without outcome relabeling.
    """
    if public.ndim != 2 or public.shape[1] != 794 or residual.shape != (len(public), 18):
        raise ValueError('Require original public794 and residual18 batches')
    if joint_scale_rad.shape != (18,) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Require the actual positive control dt and 18 scales')
    active = public[:, 770:773].sum(dim=1) > .5
    requested_rad = residual.detach().clamp(-1., 1.) * joint_scale_rad
    mean_square = requested_rad.square().mean(dim=1)
    cost = (mean_square / configuration['reference_scale_rad'] ** 2
            * configuration['weight_per_s'] * dt * active.to(mean_square.dtype))
    return cost, active, mean_square


class PreHitTeacherPriorAdapterMixin:
    """Reward wrapper; the inherited native step remains the sole dispatcher."""

    def configure_prehit_prior(self, configuration):
        self.prehit_prior = settings(configuration)
        self._prior_accumulator = torch.zeros(5, dtype=torch.float64, device=self.device)

    def step(self, residual):
        # Compute before super.step can replace observations with reset states.
        cost, active, mean_square = command_cost(
            self.last_observation['policy'], residual, self.mixer.scale,
            float(self.unwrapped.step_dt), self.prehit_prior)
        observation, reward, dones, extras = super().step(residual)
        adjusted = reward - cost
        if not torch.isfinite(adjusted).all():
            raise RuntimeError('Nonfinite prior-adjusted native training reward')
        self._prior_accumulator += torch.stack((
            cost.double().sum(), active.double().sum(),
            (mean_square.double() * active).sum(),
            reward.double().sum(), adjusted.double().sum()))
        return observation, adjusted, dones, extras

    def prehit_prior_diagnostics(self, clear=False):
        cost, samples, square, before, after = self._prior_accumulator.cpu().tolist()
        result = dict(schema=SCHEMA, cumulative_cost=cost,
                      active_control_env_samples=int(samples),
                      requested_residual_rms_rad=math.sqrt(square / samples) if samples else 0.,
                      original_reward_sum=before, learner_reward_sum=after,
                      weight_per_s=self.prehit_prior['weight_per_s'],
                      reference_scale_rad=self.prehit_prior['reference_scale_rad'],
                      public_phase_indices=[0, 1, 2], action_dispatch_unchanged=True)
        if clear:
            self._prior_accumulator.zero_()
        return result
