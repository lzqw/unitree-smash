"""Physical-time configuration and actor-critic objectives (no Isaac import).

Equations follow HybridRobotics/supermimic policy_learner.py (blob
f13731760f9d0937ad4c2f169ddf60be66c31e03). This module is an independent
implementation, not a claim of reproducing the whole G1 training pipeline.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch


def seconds_to_steps(seconds: float, dt: float, name: str = "horizon") -> int:
    if not math.isfinite(seconds) or not math.isfinite(dt) or seconds <= 0 or dt <= 0:
        raise ValueError(f"{name} and control dt must be finite and positive")
    ratio = seconds / dt
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{name}={seconds}s must be an integer multiple of dt={dt}s")
    return steps


@dataclass(frozen=True)
class Timebase:
    control_dt_s: float = 0.01
    episode_s: float = 8.0
    imagination_s: float = 0.64
    world_window_s: float = 0.16
    reference_dt_s: float = 0.02
    gamma_reference: float = 0.99
    lambda_reference: float = 0.95
    discount_mode: str = "physical_time"

    def __post_init__(self):
        for name, value in (("episode", self.episode_s), ("imagination", self.imagination_s),
                            ("world window", self.world_window_s)):
            seconds_to_steps(value, self.control_dt_s, name)
        if self.imagination_s > self.episode_s or self.world_window_s > self.episode_s:
            raise ValueError("Learning windows must not exceed the real episode")
        if not math.isfinite(self.reference_dt_s) or self.reference_dt_s <= 0:
            raise ValueError("reference_dt_s must be positive")
        if not 0 < self.gamma_reference < 1 or not 0 < self.lambda_reference <= 1:
            raise ValueError("Require 0<gamma<1 and 0<lambda<=1")
        if self.discount_mode not in ("physical_time", "per_step"):
            raise ValueError("discount_mode must be physical_time or per_step")

    @property
    def imagination_steps(self):
        return seconds_to_steps(self.imagination_s, self.control_dt_s)

    @property
    def world_steps(self):
        # K transitions require K+1 successive states. No K/K-1 ambiguity.
        return seconds_to_steps(self.world_window_s, self.control_dt_s)

    @property
    def gamma(self):
        exponent = self.control_dt_s / self.reference_dt_s if self.discount_mode == "physical_time" else 1.
        return self.gamma_reference ** exponent

    @property
    def lam(self):
        exponent = self.control_dt_s / self.reference_dt_s if self.discount_mode == "physical_time" else 1.
        return self.lambda_reference ** exponent

    def manifest(self):
        return {**asdict(self), "imagination_steps": self.imagination_steps,
                "world_transitions": self.world_steps, "world_states": self.world_steps + 1,
                "gamma_effective": self.gamma, "lambda_effective": self.lam,
                "episode_ticks": seconds_to_steps(self.episode_s, self.control_dt_s)}


def bptt_return(rewards, dones, last_value, gamma):
    """Return from the sampled initial state, not a mean of suffix returns.

    Inputs have time-major shape [T,B,1], last_value is [B,1]. The state is
    never multiplied by survival. Preserve gradients through reward/done/value.
    """
    if rewards.shape != dones.shape or rewards.ndim != 3 or last_value.shape != rewards.shape[1:]:
        raise ValueError("Expected rewards/dones [T,B,1] and value [B,1]")
    ret, survival = torch.zeros_like(last_value), torch.ones_like(last_value)
    discount = 1.0
    for reward, done in zip(rewards.unbind(0), dones.unbind(0)):
        ret = ret + discount * survival * reward
        survival = survival * (1.0 - done)
        discount *= gamma
    return ret + discount * survival * last_value


def gae_returns(values, last_value, rewards, dones, gamma, lam):
    """GAE A+V target; callers compute it on detached trajectories."""
    if values.shape != rewards.shape or dones.shape != rewards.shape or last_value.shape != values.shape[1:]:
        raise ValueError("GAE shape mismatch")
    targets = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        continuation = 1.0 - dones[t]
        delta = rewards[t] + gamma * continuation * next_value - values[t]
        advantage = delta + gamma * lam * continuation * advantage
        targets[t] = values[t] + advantage
        next_value = values[t]
    return targets


def full_range_targets(normalized, limits):
    """Decode the *final* full-racket command; never apply JumpEnv scaling."""
    return limits[..., 0] + 0.5 * (normalized.clamp(-1., 1.) + 1.) * (limits[..., 1] - limits[..., 0])
