"""Parameterised arm and tool interfaces.

The placeholder arm is intentionally labelled as an approximation until a
real manufacturer model and limits are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointCommand:
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


class ArmInterface:
    model_name = "abstract"

    def __init__(self, joint_names: tuple[str, ...], lower: np.ndarray, upper: np.ndarray, effort_limits: np.ndarray):
        self.joint_names = tuple(joint_names)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.effort_limits = np.asarray(effort_limits, dtype=np.float64)
        if not (len(self.joint_names) == len(self.lower) == len(self.upper) == len(self.effort_limits)):
            raise ValueError("arm names and limits must have the same length")
        if np.any(self.lower >= self.upper) or np.any(self.effort_limits <= 0):
            raise ValueError("arm limits must be ordered and positive")

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def command(self, desired_position: np.ndarray, current_position: np.ndarray, current_velocity: np.ndarray, kp: float = 40.0, kd: float = 2.0) -> JointCommand:
        desired_position = np.clip(np.asarray(desired_position, dtype=np.float64), self.lower, self.upper)
        current_position = np.asarray(current_position, dtype=np.float64)
        current_velocity = np.asarray(current_velocity, dtype=np.float64)
        if desired_position.shape != (self.dof,) or current_position.shape != (self.dof,) or current_velocity.shape != (self.dof,):
            raise ValueError("arm command vectors must match arm dof")
        effort = np.clip(kp * (desired_position - current_position) - kd * current_velocity, -self.effort_limits, self.effort_limits)
        return JointCommand(desired_position, np.zeros(self.dof), effort)


class PlaceholderArm(ArmInterface):
    model_name = "placeholder_6dof_unconfirmed"

    def __init__(self):
        super().__init__(
            ("arm_joint_0", "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"),
            np.array([-2.8, -1.7, -2.8, -3.1, -2.0, -3.1]),
            np.array([2.8, 1.7, 2.8, 3.1, 2.0, 3.1]),
            np.array([35.0, 35.0, 30.0, 20.0, 15.0, 10.0]),
        )


class WholeBodyInterface:
    """Normalized 12-leg + 6-arm command interface for PPO policies."""

    def __init__(self, *, action_scale: float = 0.35, arm_action_scale: float = 0.28, kp_leg: float = 35.0, kd_leg: float = 1.5, arm: ArmInterface | None = None):
        if action_scale <= 0 or arm_action_scale <= 0 or kp_leg <= 0 or kd_leg < 0:
            raise ValueError("whole-body scales and gains must be valid")
        self.arm = arm or PlaceholderArm()
        if self.arm.dof != 6:
            raise ValueError("the smash task expects a six-DoF arm")
        self.leg_lower = np.full(12, -2.5, dtype=np.float64)
        self.leg_upper = np.full(12, 2.5, dtype=np.float64)
        self.leg_effort_limits = np.full(12, 45.0, dtype=np.float64)
        self.action_scale = np.concatenate((np.full(12, action_scale), np.full(6, arm_action_scale)))
        self.kp = np.concatenate((np.full(12, kp_leg), np.full(6, 40.0)))
        self.kd = np.concatenate((np.full(12, kd_leg), np.full(6, 2.0)))
        self.lower = np.concatenate((self.leg_lower, self.arm.lower))
        self.upper = np.concatenate((self.leg_upper, self.arm.upper))
        self.effort_limits = np.concatenate((self.leg_effort_limits, self.arm.effort_limits))

    @property
    def dof(self) -> int:
        return 18

    def command(self, action: np.ndarray, current_position: np.ndarray, current_velocity: np.ndarray, nominal_position: np.ndarray | None = None) -> JointCommand:
        action = np.asarray(action, dtype=np.float64)
        current_position = np.asarray(current_position, dtype=np.float64)
        current_velocity = np.asarray(current_velocity, dtype=np.float64)
        if action.shape != (self.dof,) or current_position.shape != (self.dof,) or current_velocity.shape != (self.dof,):
            raise ValueError("whole-body vectors must have shape (18,)")
        nominal = np.zeros(self.dof, dtype=np.float64) if nominal_position is None else np.asarray(nominal_position, dtype=np.float64)
        if nominal.shape != (self.dof,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_position must be finite with shape (18,)")
        desired = np.clip(nominal + self.action_scale * np.clip(action, -1.0, 1.0), self.lower, self.upper)
        effort = np.clip(self.kp * (desired - current_position) - self.kd * current_velocity, -self.effort_limits, self.effort_limits)
        return JointCommand(desired, np.zeros(self.dof, dtype=np.float64), effort)

