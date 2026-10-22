"""Task event state and episode metric bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .goals import TaskPhase


class EpisodeEndReason(str, Enum):
    RUNNING = "running"
    CONTACT = "contact"
    TIMEOUT = "timeout"
    OUT_OF_BOUNDS = "out_of_bounds"
    UNSTABLE = "unstable"
    SUCCESS = "success"


@dataclass
class EpisodeEvents:
    contact: bool = False
    takeoff: bool = False
    airborne: bool = False
    landing: bool = False
    out_of_bounds: bool = False
    unstable: bool = False
    airborne_contact: bool = False
    event_target_reached: bool = False
    valid_return: bool = False
    attack_requirement: bool = False
    stable_recovery: bool = False
    success: bool = False
    phase: TaskPhase = TaskPhase.PREPARE
    contact_time: float | None = None
    contact_position: list[float] | None = None
    post_contact_velocity: list[float] | None = None
    landing_time: float | None = None
    recovery_steps: int = 0
    end_reason: EpisodeEndReason = EpisodeEndReason.RUNNING

    def reset(self) -> None:
        self.__dict__.update(EpisodeEvents().__dict__)

    def as_dict(self) -> dict:
        return {
            "contact": self.contact,
            "takeoff": self.takeoff,
            "airborne": self.airborne,
            "landing": self.landing,
            "out_of_bounds": self.out_of_bounds,
            "unstable": self.unstable,
            "airborne_contact": self.airborne_contact,
            "event_target_reached": self.event_target_reached,
            "valid_return": self.valid_return,
            "attack_requirement": self.attack_requirement,
            "stable_recovery": self.stable_recovery,
            "success": self.success,
            "phase": self.phase.value,
            "contact_time": self.contact_time,
            "contact_position": self.contact_position,
            "post_contact_velocity": self.post_contact_velocity,
            "landing_time": self.landing_time,
            "recovery_steps": self.recovery_steps,
            "end_reason": self.end_reason.value,
        }


@dataclass
class EpisodeMetrics:
    steps: int = 0
    contact_count: int = 0
    energy: float = 0.0
    target_error: float = 0.0
    max_tilt: float = 0.0
    takeoff_count: int = 0
    airborne_contact_count: int = 0
    valid_return_count: int = 0
    recovery_success_count: int = 0
    attack_quality: float = 0.0
    end_reason: EpisodeEndReason = EpisodeEndReason.RUNNING

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "contact_count": self.contact_count,
            "energy": self.energy,
            "target_error": self.target_error,
            "max_tilt": self.max_tilt,
            "takeoff_count": self.takeoff_count,
            "airborne_contact_count": self.airborne_contact_count,
            "valid_return_count": self.valid_return_count,
            "recovery_success_count": self.recovery_success_count,
            "attack_quality": self.attack_quality,
            "end_reason": self.end_reason.value,
        }

