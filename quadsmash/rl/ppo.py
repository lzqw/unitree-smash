"""Small NumPy PPO adapter for smoke tests and reproducible baselines.

It is intentionally a linear policy so it can run before torch/Isaac Lab are
installed. Isaac Lab training can use the same observation and action contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.997
    gae_lambda: float = 0.97
    update_epochs: int = 4
    rollout_length: int = 128
    seed: int = 7


class LinearPPO:
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.actor = self.rng.normal(0.0, 0.02, (obs_dim, action_dim))
        self.critic = np.zeros(obs_dim)
        self.log_std = np.full(action_dim, -0.5)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        logits = np.tanh(obs @ self.actor)
        if deterministic:
            return logits.astype(np.float32)
        std = np.exp(self.log_std)
        return np.clip(logits + self.rng.normal(0.0, std, logits.shape), -1.0, 1.0).astype(np.float32)

    def update(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, gamma: float = 0.99) -> dict:
        obs = np.asarray(obs, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        next_obs = np.asarray(next_obs, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.float64)
        values = obs @ self.critic
        targets = rewards + gamma * (1.0 - dones) * (next_obs @ self.critic)
        advantage = targets - values
        self.critic += self.config.learning_rate * np.mean(advantage[:, None] * obs, axis=0)
        logits = np.tanh(obs @ self.actor)
        error = actions - logits
        self.actor += self.config.learning_rate * (obs.T @ (advantage[:, None] * error)) / max(1, len(obs))
        return {"mean_reward": float(np.mean(rewards)), "mean_advantage": float(np.mean(advantage)), "actor_norm": float(np.linalg.norm(self.actor))}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, actor=self.actor, critic=self.critic, log_std=self.log_std, obs_dim=self.obs_dim, action_dim=self.action_dim)
        path.with_suffix(".json").write_text(json.dumps({"algorithm": "linear_ppo", "obs_dim": self.obs_dim, "action_dim": self.action_dim, "config": self.config.__dict__}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearPPO":
        data = np.load(path)
        agent = cls(int(data["obs_dim"]), int(data["action_dim"]))
        agent.actor, agent.critic, agent.log_std = data["actor"], data["critic"], data["log_std"]
        return agent


def _mlp_init(rng: np.random.Generator, dims: list[int]) -> list[np.ndarray]:
    return [rng.normal(0.0, np.sqrt(2.0 / max(1, dims[i])), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]


class WholeBodyPolicy:
    """Shared MLP with coupled 12-D leg and 6-D arm action heads."""

    def __init__(self, obs_dim: int, action_dim: int = 18, hidden_dims: tuple[int, ...] = (512, 256, 128), *, seed: int = 7, leg_action_scale: float = 0.35, arm_action_scale: float = 0.28):
        if action_dim < 18 or obs_dim <= 0 or not hidden_dims:
            raise ValueError("whole-body policy requires obs_dim > 0, action_dim >= 18, and hidden layers")
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.hidden_dims = tuple(hidden_dims)
        self.shared = _mlp_init(self.rng, [self.obs_dim, *self.hidden_dims])
        last = self.hidden_dims[-1]
        self.leg_head = self.rng.normal(0.0, 0.02, (last, 12))
        self.arm_head = self.rng.normal(0.0, 0.02, (last, self.action_dim - 12))
        self.leg_bias = np.zeros(12)
        self.arm_bias = np.zeros(self.action_dim - 12)
        self.log_std = np.full(self.action_dim, -1.2, dtype=np.float64)
        self.action_scales = np.concatenate((np.full(12, leg_action_scale), np.full(self.action_dim - 12, arm_action_scale)))

    def _forward(self, obs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        h = np.asarray(obs, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.obs_dim:
            raise ValueError(f"obs must have shape (batch, {self.obs_dim})")
        activations = [h]
        for weight in self.shared:
            h = np.tanh(h @ weight)
            activations.append(h)
        leg = np.tanh(h @ self.leg_head + self.leg_bias)
        arm = np.tanh(h @ self.arm_head + self.arm_bias)
        return np.concatenate((leg, arm), axis=1), activations

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        return self._forward(obs)[0].astype(np.float32)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        mean = self.mean_action(obs).astype(np.float64)
        if deterministic:
            return mean.astype(np.float32)
        action = mean + self.rng.normal(0.0, np.exp(self.log_std), mean.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def log_prob(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        mean = self.mean_action(obs).astype(np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        std = np.exp(self.log_std)
        return np.sum(-0.5 * (((actions - mean) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)), axis=1)

    def _apply_gradient(self, obs: np.ndarray, grad_mean: np.ndarray, learning_rate: float) -> None:
        mean, activations = self._forward(obs)
        grad_leg = grad_mean[:, :12] * (1.0 - mean[:, :12] ** 2)
        grad_arm = grad_mean[:, 12:] * (1.0 - mean[:, 12:] ** 2)
        h = activations[-1]
        grad_h = grad_leg @ self.leg_head.T + grad_arm @ self.arm_head.T
        self.leg_head -= learning_rate * (h.T @ grad_leg) / max(1, len(obs))
        self.leg_bias -= learning_rate * np.mean(grad_leg, axis=0)
        self.arm_head -= learning_rate * (h.T @ grad_arm) / max(1, len(obs))
        self.arm_bias -= learning_rate * np.mean(grad_arm, axis=0)
        for index in range(len(self.shared) - 1, -1, -1):
            grad_h *= 1.0 - activations[index + 1] ** 2
            self.shared[index] -= learning_rate * (activations[index].T @ grad_h) / max(1, len(obs))
            if index:
                grad_h = grad_h @ self.shared[index].T


class PrivilegedCritic:
    """Value function that may consume simulator-only state."""

    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...] = (256, 128), *, seed: int = 7):
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)
        self.weights = _mlp_init(self.rng, [self.obs_dim, *hidden_dims, 1])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float64)
        for index, weight in enumerate(self.weights):
            h = h @ weight
            if index < len(self.weights) - 1:
                h = np.tanh(h)
        return h[:, 0]

    def fit_step(self, obs: np.ndarray, target: np.ndarray, learning_rate: float) -> float:
        # A stable one-step TD value update; full Isaac Lab training can swap
        # this for its minibatch critic without changing the task contract.
        pred = self.predict(obs)
        error = pred - np.asarray(target, dtype=np.float64)
        h = np.asarray(obs, dtype=np.float64)
        activations = [h]
        for weight in self.weights[:-1]:
            h = np.tanh(h @ weight)
            activations.append(h)
        grad = (2.0 * error / max(1, len(obs)))[:, None]
        self.weights[-1] -= learning_rate * activations[-1].T @ grad
        return float(np.mean(error * error))


class WholeBodyPPO:
    """NumPy PPO baseline with a shared trunk and separate leg/arm heads.

    It is intentionally small but implements clipped probability-ratio updates,
    a privileged critic hook, and correct terminal bootstrapping for the
    dependency-light backend.
    """

    def __init__(self, obs_dim: int, action_dim: int = 18, config: PPOConfig | None = None, *, critic_obs_dim: int | None = None):
        self.config = config or PPOConfig()
        self.policy = WholeBodyPolicy(obs_dim, action_dim, seed=self.config.seed)
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.critic = PrivilegedCritic(self.critic_obs_dim, seed=self.config.seed + 1)
        self.obs_dim, self.action_dim = obs_dim, action_dim

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.policy.act(obs, deterministic=deterministic)

    def update(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray, gamma: float | None = None, *, privileged_obs: np.ndarray | None = None, next_privileged_obs: np.ndarray | None = None) -> dict:
        obs = np.asarray(obs, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.float64).reshape(-1)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or actions.shape != (len(obs), self.action_dim) or len(rewards) != len(obs):
            raise ValueError("PPO batch has incompatible shapes")
        discount = self.config.gamma if gamma is None else float(gamma)
        critic_obs = obs if privileged_obs is None else np.asarray(privileged_obs, dtype=np.float64)
        next_critic_obs = next_obs if next_privileged_obs is None else np.asarray(next_privileged_obs, dtype=np.float64)
        if critic_obs.ndim != 2 or critic_obs.shape[0] != len(obs) or critic_obs.shape[1] != self.critic_obs_dim or next_critic_obs.shape != critic_obs.shape:
            raise ValueError("privileged critic observations have incompatible shapes")
        values = self.critic.predict(critic_obs)
        next_values = self.critic.predict(next_critic_obs)
        targets = rewards + discount * (1.0 - dones) * next_values
        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_prob = self.policy.log_prob(obs, actions)
        for _ in range(max(1, self.config.update_epochs)):
            mean = self.policy.mean_action(obs).astype(np.float64)
            log_prob = self.policy.log_prob(obs, actions)
            ratio = np.exp(np.clip(log_prob - old_log_prob, -20.0, 20.0))
            clipped = np.clip(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            use_ratio = ((advantages >= 0) & (ratio <= clipped)) | ((advantages < 0) & (ratio >= clipped))
            score = np.where(use_ratio, ratio * advantages, 0.0)
            grad_log_prob = (actions - mean) / (np.exp(self.policy.log_std) ** 2)
            grad_mean = -score[:, None] * grad_log_prob
            self.policy._apply_gradient(obs, grad_mean, self.config.learning_rate)
        critic_loss = self.critic.fit_step(critic_obs, targets, self.config.learning_rate)
        return {"mean_reward": float(np.mean(rewards)), "mean_advantage": float(np.mean(advantages)), "critic_loss": critic_loss, "clip_fraction": float(np.mean(np.abs(ratio - clipped) > 1e-8))}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"shared_{i}": value for i, value in enumerate(self.policy.shared)}
        arrays.update({"leg_head": self.policy.leg_head, "arm_head": self.policy.arm_head, "leg_bias": self.policy.leg_bias, "arm_bias": self.policy.arm_bias, "log_std": self.policy.log_std})
        arrays.update({f"critic_{i}": value for i, value in enumerate(self.critic.weights)})
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({"algorithm": "whole_body_ppo", "obs_dim": self.obs_dim, "action_dim": self.action_dim, "critic_obs_dim": self.critic_obs_dim, "config": self.config.__dict__}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WholeBodyPPO":
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        config = PPOConfig(**{key: value for key, value in metadata.get("config", {}).items() if key in PPOConfig.__dataclass_fields__})
        agent = cls(int(metadata["obs_dim"]), int(metadata["action_dim"]), config, critic_obs_dim=int(metadata.get("critic_obs_dim", metadata["obs_dim"])))
        data = np.load(path)
        agent.policy.shared = [data[f"shared_{i}"] for i in range(len(agent.policy.shared))]
        agent.policy.leg_head, agent.policy.arm_head = data["leg_head"], data["arm_head"]
        agent.policy.leg_bias, agent.policy.arm_bias, agent.policy.log_std = data["leg_bias"], data["arm_bias"], data["log_std"]
        agent.critic.weights = [data[f"critic_{i}"] for i in range(len(agent.critic.weights))]
        return agent

