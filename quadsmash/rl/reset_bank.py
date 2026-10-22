"""Physically valid intermediate-state reset bank.

States are captured only at episode boundaries or from validated rollouts.  A
bank entry is never injected mid-rollout and is not treated as old PPO data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass
class ResetState:
    values: dict[str, np.ndarray]
    phase: str = "prepare"
    source: str = "rollout"

    def copy(self) -> "ResetState":
        return ResetState({key: value.copy() for key, value in self.values.items()}, self.phase, self.source)


class ReachableStateBank:
    """Bounded reservoir of complete, finite dynamic states."""

    REQUIRED = ("joint_pos", "joint_vel", "base_pos", "base_vel", "base_ang_vel", "shuttle_pos", "shuttle_vel", "prev_action")

    def __init__(self, capacity: int = 4096, seed: int = 7):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._states: list[ResetState] = []

    @property
    def size(self) -> int:
        return len(self._states)

    def add(self, state: Mapping[str, np.ndarray], *, phase: str = "prepare", source: str = "rollout") -> None:
        values = {}
        for name in self.REQUIRED:
            if name not in state:
                raise ValueError(f"reset state missing {name}")
            value = np.asarray(state[name], dtype=np.float64)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"reset state {name} must be a finite vector")
            values[name] = value.copy()
        for name in ("base_quat", "shuttle_quat", "shuttle_omega", "phase_index"):
            if name in state:
                value = np.asarray(state[name], dtype=np.float64)
                if value.ndim != 1 or not np.all(np.isfinite(value)):
                    raise ValueError(f"reset state {name} must be a finite vector")
                values[name] = value.copy()
        candidate = ResetState(values, str(phase), str(source))
        if self.size < self.capacity:
            self._states.append(candidate)
        else:
            # Reservoir replacement keeps early and late curriculum states.
            index = int(self.rng.integers(0, self.size + 1))
            if index < self.capacity:
                self._states[index] = candidate

    def add_from_environment(self, env, *, index: int = 0, phase: str | None = None, source: str = "rollout") -> None:
        state = env.capture_reset_state(index)
        actual_phase = phase or env.phase[index].value
        self.add(state, phase=actual_phase, source=source)

    def sample(self, *, phase: str | None = None) -> ResetState:
        if not self._states:
            raise LookupError("reset bank is empty")
        candidates = self._states if phase is None else [state for state in self._states if state.phase == phase]
        if not candidates:
            raise LookupError(f"reset bank has no state for phase {phase!r}")
        return candidates[int(self.rng.integers(0, len(candidates)))].copy()

    def states(self) -> tuple[ResetState, ...]:
        return tuple(state.copy() for state in self._states)

