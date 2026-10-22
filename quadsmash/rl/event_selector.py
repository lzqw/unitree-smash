"""Recovery-aware contact event selection.

The selector is deliberately separated from low-level PPO: it ranks a small
set of physically screened events using a predictor trained on current-policy
rollouts.  The predictor is a capability estimate, not a formal safety proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..tasks.goals import HitEventGoal


@dataclass(frozen=True)
class HitEventCandidate:
    goal: HitEventGoal
    attack_quality: float
    effort_cost: float
    source_time: float

    def feature(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        return np.concatenate((state, self.goal.as_vector(self.source_time), [self.attack_quality, self.effort_cost]))


class RecoveryFeasibilityPredictor:
    """Online logistic capability model with uncertainty estimate."""

    def __init__(self, feature_dim: int, seed: int = 7, prior_strength: float = 1.0):
        if feature_dim <= 0 or prior_strength <= 0:
            raise ValueError("feature_dim and prior_strength must be positive")
        self.feature_dim = int(feature_dim)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(self.feature_dim, dtype=np.float64)
        self.bias = 0.0
        self.count = 0
        self.prior_strength = float(prior_strength)

    def _logit(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must end in {self.feature_dim}")
        return np.clip(features @ self.weights + self.bias, -30.0, 30.0)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return probability and a conservative standard-error proxy."""
        features = np.asarray(features, dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-self._logit(features)))
        uncertainty = 1.0 / np.sqrt(self.prior_strength + max(1, self.count))
        return probabilities, np.full(probabilities.shape, uncertainty, dtype=np.float64)

    def fit(self, features: np.ndarray, labels: np.ndarray, *, epochs: int = 4, learning_rate: float = 0.05) -> dict:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim or len(labels) != len(features):
            raise ValueError("features and labels have incompatible shapes")
        if len(labels) == 0 or not np.all(np.isfinite(features)) or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("training data must be finite, non-empty, and binary")
        for _ in range(max(1, epochs)):
            probs, _ = self.predict(features)
            error = probs - labels
            self.weights -= learning_rate * (features.T @ error / len(labels) + self.weights / self.prior_strength)
            self.bias -= learning_rate * float(np.mean(error))
        self.count += len(labels)
        probs, _ = self.predict(features)
        return {"samples": int(len(labels)), "loss": float(-np.mean(labels * np.log(probs + 1e-8) + (1.0 - labels) * np.log(1.0 - probs + 1e-8)))}


def physical_candidate_filter(state: np.ndarray, candidates: list[HitEventCandidate], *, max_time: float = 2.0, max_effort: float = 10.0) -> list[HitEventCandidate]:
    """Apply cheap necessary-condition filters before policy capability checks."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(state)):
        raise ValueError("state must be finite")
    return [candidate for candidate in candidates if 0.0 <= candidate.goal.time <= max_time and candidate.effort_cost <= max_effort and np.all(np.isfinite(candidate.goal.position))]


def candidates_from_trajectory(position: np.ndarray, velocity: np.ndarray, times: np.ndarray, *, normal: np.ndarray | None = None, attack_velocity: np.ndarray | None = None, effort_scale: float = 1.0) -> list[HitEventCandidate]:
    """Create contact candidates on a predicted constant-velocity trajectory."""
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("position and velocity must be finite 3-vectors")
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0) or not np.isfinite(effort_scale) or effort_scale < 0:
        raise ValueError("times must be finite and non-negative")
    normal = np.asarray([1.0, 0.0, 0.0] if normal is None else normal, dtype=np.float64)
    attack_velocity = np.asarray([4.0, 0.0, 0.0] if attack_velocity is None else attack_velocity, dtype=np.float64)
    effort = float(np.linalg.norm(attack_velocity - velocity) * effort_scale)
    return [HitEventCandidate(HitEventGoal(float(time), position + velocity * time, normal, attack_velocity), float(np.clip(np.linalg.norm(attack_velocity) / 8.0, 0.0, 1.0)), effort, float(time)) for time in times]


def select_event(state: np.ndarray, candidates: list[HitEventCandidate], predictor: RecoveryFeasibilityPredictor | None = None, *, min_probability: float = 0.55, uncertainty_scale: float = 1.0, effort_weight: float = 0.15) -> HitEventCandidate | None:
    """Choose the highest attack-value event with a conservative capability bound."""
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be in [0,1]")
    candidates = physical_candidate_filter(state, candidates)
    if not candidates:
        return None
    if predictor is None:
        return max(candidates, key=lambda candidate: candidate.attack_quality - effort_weight * candidate.effort_cost)
    features = np.stack([candidate.feature(state) for candidate in candidates])
    probabilities, uncertainty = predictor.predict(features)
    lower = probabilities - uncertainty_scale * uncertainty
    feasible = lower >= min_probability
    if not np.any(feasible):
        return None
    scores = np.array([candidate.attack_quality - effort_weight * candidate.effort_cost for candidate in candidates])
    scores[~feasible] = -np.inf
    return candidates[int(np.argmax(scores))]

