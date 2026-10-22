"""Event-conditioned task goals and curriculum definitions.

The numerical environment uses the same goal contract as the future Isaac Lab
task: a policy is asked to achieve a racket event at a particular time, pose,
and velocity, then recover.  Keeping this contract in a small dependency-free
module makes it possible to train and test the task before the final assets are
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class TaskPhase(str, Enum):
    PREPARE = "prepare"
    TAKEOFF = "takeoff"
    AIRBORNE = "airborne"
    POST_CONTACT = "post_contact"
    RECOVERY = "recovery"
    COMPLETE = "complete"

    @classmethod
    def ordered(cls) -> tuple["TaskPhase", ...]:
        return (cls.PREPARE, cls.TAKEOFF, cls.AIRBORNE, cls.POST_CONTACT, cls.RECOVERY)


class CurriculumStage(str, Enum):
    """Difficulty levels from the implementation plan."""

    JUMP_RECOVERY = "jump_recovery"
    VIRTUAL_EVENT = "virtual_event"
    REAL_CONTACT = "real_contact"
    SMASH_AND_RECOVER = "smash_and_recover"
    DOMAIN_RANDOMIZED = "domain_randomized"


@dataclass(frozen=True)
class HitEventGoal:
    """A time-conditioned racket contact target in the world frame."""

    time: float
    position: np.ndarray
    normal: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("goal time must be finite")
        for name in ("position", "normal", "velocity"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"goal {name} must be a finite 3-vector")
            object.__setattr__(self, name, value)
        norm = float(np.linalg.norm(self.normal))
        if norm < 1e-9:
            raise ValueError("goal normal must be non-zero")
        object.__setattr__(self, "normal", self.normal / norm)

    def remaining(self, now: float) -> float:
        return float(self.time - now)

    def as_vector(self, now: float) -> np.ndarray:
        """Return the policy representation using remaining time, not absolute time."""
        return np.concatenate(([self.remaining(now)], self.position, self.normal, self.velocity)).astype(np.float32)


def phase_one_hot(phase: TaskPhase) -> np.ndarray:
    values = np.zeros(len(TaskPhase.ordered()), dtype=np.float32)
    try:
        values[TaskPhase.ordered().index(phase)] = 1.0
    except ValueError:
        # COMPLETE is represented by the recovery component for a fixed-size
        # policy observation.
        values[-1] = 1.0
    return values


@dataclass(frozen=True)
class CurriculumConfig:
    """Difficulty knobs for the staged task."""

    stage: CurriculumStage = CurriculumStage.REAL_CONTACT
    time_tolerance: float = 0.12
    position_tolerance: float = 0.18
    normal_tolerance: float = 0.65
    velocity_tolerance: float = 3.0
    require_real_contact: bool = True
    require_valid_return: bool = False
    require_attack: bool = False
    require_recovery: bool = True

    @classmethod
    def for_stage(cls, stage: CurriculumStage | str) -> "CurriculumConfig":
        stage = CurriculumStage(stage)
        if stage is CurriculumStage.JUMP_RECOVERY:
            return cls(stage, require_real_contact=False, require_valid_return=False, require_attack=False)
        if stage is CurriculumStage.VIRTUAL_EVENT:
            return cls(stage, time_tolerance=0.18, position_tolerance=0.25, normal_tolerance=0.9, velocity_tolerance=5.0, require_real_contact=False, require_valid_return=False, require_attack=False)
        if stage is CurriculumStage.SMASH_AND_RECOVER:
            return cls(stage, time_tolerance=0.10, position_tolerance=0.14, normal_tolerance=0.5, velocity_tolerance=2.5, require_real_contact=True, require_valid_return=True, require_attack=True)
        if stage is CurriculumStage.DOMAIN_RANDOMIZED:
            return cls(stage, time_tolerance=0.08, position_tolerance=0.12, normal_tolerance=0.45, velocity_tolerance=2.0, require_real_contact=True, require_valid_return=True, require_attack=True)
        return cls(stage)


@dataclass
class CurriculumScheduler:
    """Promote difficulty only after fixed validation cases are mastered."""

    stages: tuple[CurriculumStage, ...] = (
        CurriculumStage.JUMP_RECOVERY,
        CurriculumStage.VIRTUAL_EVENT,
        CurriculumStage.REAL_CONTACT,
        CurriculumStage.SMASH_AND_RECOVER,
        CurriculumStage.DOMAIN_RANDOMIZED,
    )
    index: int = 0
    success_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not self.stages or not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("curriculum stages and success threshold are invalid")
        self.index = int(np.clip(self.index, 0, len(self.stages) - 1))

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.index]

    @property
    def config(self) -> CurriculumConfig:
        return CurriculumConfig.for_stage(self.stage)

    def update(self, validation_success_rate: float) -> bool:
        if not np.isfinite(validation_success_rate) or not 0.0 <= validation_success_rate <= 1.0:
            raise ValueError("validation_success_rate must be in [0,1]")
        if validation_success_rate >= self.success_threshold and self.index < len(self.stages) - 1:
            self.index += 1
            return True
        return False
