"""Deterministic low-level skill baselines used before whole-body PPO."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SkillResult:
    command: np.ndarray
    event: str
    success: bool
    metrics: dict


class StandSkill:
    name = "stand"

    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints

    def step(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> SkillResult:
        target = np.zeros(self.num_joints)
        command = np.clip(2.0 * (target - joint_pos) - 0.2 * joint_vel, -1.0, 1.0)
        return SkillResult(command, "standing", bool(np.max(np.abs(joint_pos)) < 0.1), {"joint_error": float(np.linalg.norm(joint_pos - target))})


class ArmSwingSkill:
    name = "arm_swing"

    def __init__(self, num_joints: int = 6):
        self.num_joints = num_joints

    def step(self, phase: float, joint_pos: np.ndarray) -> SkillResult:
        phase = float(np.clip(phase, 0.0, 1.0))
        target = np.zeros(self.num_joints)
        target[0] = -0.8 * np.sin(np.pi * phase)
        target[1] = 0.5 * np.sin(np.pi * phase)
        target[2] = 0.6 * np.sin(np.pi * phase)
        command = np.clip(3.0 * (target - joint_pos), -1.0, 1.0)
        return SkillResult(command, "swinging" if phase < 1.0 else "swing_complete", phase >= 1.0, {"phase": phase})


class LocomotionSkill:
    name = "locomotion"

    def step(self, command_xy_yaw: np.ndarray, base_velocity: np.ndarray) -> SkillResult:
        command_xy_yaw = np.asarray(command_xy_yaw, dtype=np.float64)
        base_velocity = np.asarray(base_velocity, dtype=np.float64)
        error = command_xy_yaw[:2] - base_velocity[:2]
        return SkillResult(np.clip(error, -1.0, 1.0), "moving", bool(np.linalg.norm(error) < 0.1), {"velocity_error": float(np.linalg.norm(error))})


class JumpRecoverySkill:
    name = "jump_recovery"

    def step(self, base_height: float, vertical_velocity: float) -> SkillResult:
        if base_height > 0.42 and vertical_velocity > 0:
            event = "takeoff"
        elif base_height > 0.25:
            event = "airborne"
        else:
            event = "landing_recovery"
        return SkillResult(np.zeros(4), event, event == "landing_recovery", {"base_height": float(base_height), "vertical_velocity": float(vertical_velocity)})

