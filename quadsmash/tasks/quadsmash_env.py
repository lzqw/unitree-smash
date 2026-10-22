"""Dependency-light event-conditioned whole-body smash environment.

This backend deliberately keeps the physics simple enough for CI, while
preserving the contracts needed by an Isaac Lab implementation: one 18-D
whole-body action, a time-conditioned racket event, phase transitions from
physical events, real collision handling, and a post-hit recovery window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..physics.collision import RacketGeometry, resolve_racket_collision, swept_sphere_plane_contact
from ..physics.shuttlecock import ShuttlecockModel, ShuttlecockParams
from .events import EpisodeEndReason, EpisodeEvents, EpisodeMetrics
from .goals import CurriculumConfig, CurriculumStage, HitEventGoal, TaskPhase, phase_one_hot
from .observations import build_observation, observation_size
from .rewards import compute_reward, event_tracking_reward, outcome_success


@dataclass
class QuadSmashEnvConfig:
    num_envs: int = 32
    num_joints: int = 18
    dt: float = 0.02
    episode_seconds: float = 4.0
    seed: int = 7
    target: tuple[float, float, float] = (2.5, 0.0, 0.1)
    action_scale: float = 0.35
    arm_action_scale: float = 0.28
    curriculum_stage: CurriculumStage | str = CurriculumStage.REAL_CONTACT
    goal_time: float = 0.85
    goal_position: tuple[float, float, float] = (0.70, 0.0, 1.0)
    goal_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    goal_velocity: tuple[float, float, float] = (4.0, 0.0, 0.0)
    racket_radius: float = 0.11
    recovery_steps: int = 10
    nominal_base_height: float = 0.34
    airborne_height: float = 0.40
    max_tilt_rate: float = 8.0

    @property
    def max_steps(self) -> int:
        return int(math.ceil(self.episode_seconds / self.dt))

    @property
    def curriculum(self) -> CurriculumConfig:
        return CurriculumConfig.for_stage(self.curriculum_stage)


class _Box:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype


class QuadSmashEnv:
    """Parallel task API matching the Isaac Lab integration boundary."""

    metadata = {"render_modes": []}

    def __init__(self, config: QuadSmashEnvConfig | None = None):
        self.cfg = config or QuadSmashEnvConfig()
        if self.cfg.num_envs <= 0 or self.cfg.num_joints < 18 or self.cfg.dt <= 0:
            raise ValueError("num_envs, num_joints (>=18), and dt must be positive")
        if self.cfg.dt > 0.05 or self.cfg.episode_seconds <= 0:
            raise ValueError("dt must be <= 0.05 and episode_seconds positive")
        if self.cfg.recovery_steps <= 0:
            raise ValueError("recovery_steps must be positive")
        self.num_envs = self.cfg.num_envs
        self.num_joints = self.cfg.num_joints
        self.num_leg_joints = 12
        self.num_arm_joints = self.num_joints - self.num_leg_joints
        self.observation_space = _Box(-np.inf, np.inf, (observation_size(self.num_joints),))
        self.action_space = _Box(-1.0, 1.0, (self.num_joints,))
        self.rng = np.random.default_rng(self.cfg.seed)
        self.model = ShuttlecockModel(ShuttlecockParams(drag_coefficient=0.12))
        self.target_pos = np.tile(np.asarray(self.cfg.target, dtype=np.float64), (self.num_envs, 1))
        self.episode_events = [EpisodeEvents() for _ in range(self.num_envs)]
        self.episode_metrics = [EpisodeMetrics() for _ in range(self.num_envs)]
        self.goals: tuple[HitEventGoal, ...] = tuple()
        self._allocate()
        self.reset(seed=self.cfg.seed)

    def _allocate(self) -> None:
        n, j = self.num_envs, self.num_joints
        self.joint_pos = np.zeros((n, j), dtype=np.float64)
        self.joint_vel = np.zeros((n, j), dtype=np.float64)
        self.base_pos = np.zeros((n, 3), dtype=np.float64)
        self.base_vel = np.zeros((n, 3), dtype=np.float64)
        self.base_ang_vel = np.zeros((n, 3), dtype=np.float64)
        self.base_quat = np.zeros((n, 4), dtype=np.float64)
        self.shuttle_pos = np.zeros((n, 3), dtype=np.float64)
        self.shuttle_vel = np.zeros((n, 3), dtype=np.float64)
        self.shuttle_quat = np.zeros((n, 4), dtype=np.float64)
        self.shuttle_omega = np.zeros((n, 3), dtype=np.float64)
        self.racket_pos = np.zeros((n, 3), dtype=np.float64)
        self.racket_vel = np.zeros((n, 3), dtype=np.float64)
        self.racket_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        self.foot_contacts = np.zeros((n, 4), dtype=bool)
        self.phase = [TaskPhase.PREPARE for _ in range(n)]
        self.step_count = np.zeros(n, dtype=np.int64)
        self.recovery_count = np.zeros(n, dtype=np.int64)
        self.done = np.zeros(n, dtype=bool)
        self.contact_pulse = np.zeros(n, dtype=np.float64)
        self.success_pulse = np.zeros(n, dtype=np.float64)
        self.virtual_event_pulse = np.zeros(n, dtype=np.float64)
        self.attack_quality = np.zeros(n, dtype=np.float64)
        self.prev_action = np.zeros((n, j), dtype=np.float64)
        self.last_obs = np.zeros((n, observation_size(j)), dtype=np.float32)

    @property
    def curriculum_stage(self) -> CurriculumStage:
        return CurriculumStage(self.cfg.curriculum_stage)

    def set_curriculum_stage(self, stage: CurriculumStage | str) -> None:
        """Change difficulty between episodes, never in the middle of one."""
        self.cfg.curriculum_stage = CurriculumStage(stage)

    def _make_goals(self) -> tuple[HitEventGoal, ...]:
        base_pos = np.asarray(self.cfg.goal_position, dtype=np.float64)
        normal = np.asarray(self.cfg.goal_normal, dtype=np.float64)
        velocity = np.asarray(self.cfg.goal_velocity, dtype=np.float64)
        goals = []
        for _ in range(self.num_envs):
            if self.cfg.curriculum_stage == CurriculumStage.DOMAIN_RANDOMIZED or self.cfg.curriculum_stage == CurriculumStage.DOMAIN_RANDOMIZED.value:
                pos = base_pos + self.rng.normal(0.0, [0.04, 0.06, 0.04])
                t = self.cfg.goal_time + float(self.rng.normal(0.0, 0.04))
            else:
                pos, t = base_pos.copy(), self.cfg.goal_time
            goals.append(HitEventGoal(max(0.05, t), pos, normal, velocity))
        return tuple(goals)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.num_envs
        self.joint_pos.fill(0.0)
        self.joint_vel.fill(0.0)
        self.base_pos[:] = np.array([0.0, 0.0, self.cfg.nominal_base_height])
        self.base_vel.fill(0.0)
        self.base_ang_vel.fill(0.0)
        self.base_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.shuttle_pos[:] = np.array([1.0, 0.0, 1.5]) + self.rng.normal(0.0, 0.03, (n, 3))
        self.shuttle_pos[:, 2] = np.maximum(self.shuttle_pos[:, 2], 0.5)
        self.shuttle_vel[:] = np.array([-1.2, 0.0, -0.8]) + self.rng.normal(0.0, 0.05, (n, 3))
        self.shuttle_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.shuttle_omega.fill(0.0)
        self.foot_contacts[:] = True
        self.step_count.fill(0)
        self.recovery_count.fill(0)
        self.done.fill(False)
        self.contact_pulse.fill(0.0)
        self.success_pulse.fill(0.0)
        self.virtual_event_pulse.fill(0.0)
        self.attack_quality.fill(0.0)
        self.prev_action.fill(0.0)
        self.phase = [TaskPhase.PREPARE for _ in range(n)]
        self.goals = self._make_goals()
        for events, metrics in zip(self.episode_events, self.episode_metrics):
            events.reset()
            metrics.__dict__.update(EpisodeMetrics().__dict__)
        if options and options.get("state") is not None:
            self._restore_reset_state(options["state"])
        self._update_racket_state()
        self.last_obs = self.get_observations()
        return self.last_obs.copy(), {"seed": seed, "num_envs": self.num_envs, "curriculum_stage": CurriculumStage(self.cfg.curriculum_stage).value}

    def reset_indices(self, indices: np.ndarray | list[int], *, seed: int | None = None) -> np.ndarray:
        """Reset only selected parallel environments and return their observations."""
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return np.empty((0, self.observation_space.shape[0]), dtype=np.float32)
        if np.any(selected < 0) or np.any(selected >= self.num_envs) or len(np.unique(selected)) != len(selected):
            raise ValueError("reset indices must be unique and within num_envs")
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self.joint_pos[selected] = 0.0
        self.joint_vel[selected] = 0.0
        self.base_pos[selected] = np.array([0.0, 0.0, self.cfg.nominal_base_height])
        self.base_vel[selected] = 0.0
        self.base_ang_vel[selected] = 0.0
        self.base_quat[selected] = np.array([1.0, 0.0, 0.0, 0.0])
        self.shuttle_pos[selected] = np.array([1.0, 0.0, 1.5]) + rng.normal(0.0, 0.03, (len(selected), 3))
        self.shuttle_pos[selected, 2] = np.maximum(self.shuttle_pos[selected, 2], 0.5)
        self.shuttle_vel[selected] = np.array([-1.2, 0.0, -0.8]) + rng.normal(0.0, 0.05, (len(selected), 3))
        self.shuttle_quat[selected] = np.array([1.0, 0.0, 0.0, 0.0])
        self.shuttle_omega[selected] = 0.0
        self.foot_contacts[selected] = True
        self.step_count[selected] = 0
        self.recovery_count[selected] = 0
        self.done[selected] = False
        self.contact_pulse[selected] = 0.0
        self.success_pulse[selected] = 0.0
        self.virtual_event_pulse[selected] = 0.0
        self.attack_quality[selected] = 0.0
        self.prev_action[selected] = 0.0
        for index in selected:
            self.episode_events[int(index)].reset()
            self.episode_metrics[int(index)].__dict__.update(EpisodeMetrics().__dict__)
        goals = list(self.goals)
        base_goal = np.asarray(self.cfg.goal_position, dtype=np.float64)
        for index in selected:
            if self.curriculum_stage is CurriculumStage.DOMAIN_RANDOMIZED:
                position = base_goal + rng.normal(0.0, [0.04, 0.06, 0.04])
                goal_time = max(0.05, self.cfg.goal_time + float(rng.normal(0.0, 0.04)))
            else:
                position, goal_time = base_goal.copy(), self.cfg.goal_time
            goals[int(index)] = HitEventGoal(goal_time, position, self.cfg.goal_normal, self.cfg.goal_velocity)
            self.phase[int(index)] = TaskPhase.PREPARE
        self.goals = tuple(goals)
        self._update_racket_state()
        selected_obs = self.get_observations()[selected].copy()
        self.last_obs[selected] = selected_obs
        return selected_obs

    def _restore_reset_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore one broadcastable, physically captured reset state."""
        for name in ("joint_pos", "joint_vel", "base_pos", "base_vel", "base_ang_vel", "base_quat", "shuttle_pos", "shuttle_vel", "shuttle_quat", "shuttle_omega", "prev_action"):
            if name in state:
                value = np.asarray(state[name], dtype=np.float64)
                target = getattr(self, name)
                if value.shape == target.shape[1:]:
                    value = np.broadcast_to(value, target.shape)
                if value.shape != target.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"reset state {name} has invalid shape or values")
                target[:] = value
        if "phase_index" in state:
            phase_index = int(np.asarray(state["phase_index"]).reshape(-1)[0])
            ordered = TaskPhase.ordered()
            if not 0 <= phase_index < len(ordered):
                raise ValueError("reset phase_index is invalid")
            self.phase = [ordered[phase_index] for _ in range(self.num_envs)]
            for event in self.episode_events:
                event.phase = ordered[phase_index]
                event.takeoff = phase_index >= 1
                event.airborne = phase_index in (1, 2)
                event.contact = phase_index >= 3
        self.foot_contacts[:] = self.base_pos[:, 2:3] <= self.cfg.nominal_base_height + 0.01

    def capture_reset_state(self, index: int = 0) -> dict[str, np.ndarray]:
        """Capture complete dynamic state for a future episode initialization."""
        if not 0 <= index < self.num_envs:
            raise IndexError(index)
        return {name: getattr(self, name)[index].copy() for name in ("joint_pos", "joint_vel", "base_pos", "base_vel", "base_ang_vel", "base_quat", "shuttle_pos", "shuttle_vel", "shuttle_quat", "shuttle_omega", "prev_action")} | {"phase_index": np.array([TaskPhase.ordered().index(self.phase[index]) if self.phase[index] in TaskPhase.ordered() else 4], dtype=np.float64)}

    def _update_racket_state(self) -> None:
        """A compact placeholder kinematic map for the 6-DoF arm interface."""
        arm = self.joint_pos[:, self.num_leg_joints : self.num_leg_joints + 6]
        arm_vel = self.joint_vel[:, self.num_leg_joints : self.num_leg_joints + 6]
        self.racket_pos = self.base_pos + np.stack((0.65 + 0.10 * np.tanh(arm[:, 0]), 0.12 * np.tanh(arm[:, 1]), 0.55 + 0.16 * np.tanh(arm[:, 2])), axis=1)
        self.racket_vel = self.base_vel + np.stack((0.10 * (1.0 - np.tanh(arm[:, 0]) ** 2) * arm_vel[:, 0], 0.12 * (1.0 - np.tanh(arm[:, 1]) ** 2) * arm_vel[:, 1], 0.16 * (1.0 - np.tanh(arm[:, 2]) ** 2) * arm_vel[:, 2]), axis=1)
        normal = np.stack((np.ones(self.num_envs), np.tanh(arm[:, 4]), np.tanh(arm[:, 5])), axis=1)
        self.racket_normal = normal / np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-9)

    def get_observations(self) -> np.ndarray:
        now = self.step_count.astype(np.float64) * self.cfg.dt
        flags = np.array([[e.takeoff, e.contact, e.landing] for e in self.episode_events], dtype=np.float64)
        return build_observation(self.joint_pos, self.joint_vel, self.base_pos, self.base_vel, self.base_quat, self.shuttle_pos, self.shuttle_vel, self.shuttle_quat, self.shuttle_omega, self.target_pos, self.foot_contacts, now=now, goals=self.goals, racket_pos=self.racket_pos, racket_vel=self.racket_vel, racket_normal=self.racket_normal, phases=self.phase, event_flags=flags, previous_actions=self.prev_action, base_ang_vel=self.base_ang_vel)

    def _tracking_terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now = self.step_count.astype(np.float64) * self.cfg.dt
        goals = np.array([goal.as_vector(float(now[i])) for i, goal in enumerate(self.goals)], dtype=np.float64)
        time_error = goals[:, 0]
        position_error = np.sum((self.racket_pos - goals[:, 1:4]) ** 2, axis=1)
        normal_alignment = np.sum(self.racket_normal * goals[:, 4:7], axis=1)
        velocity_error = np.sum((self.racket_vel - goals[:, 7:10]) ** 2, axis=1)
        return time_error, position_error, normal_alignment, velocity_error

    def get_rewards(self) -> np.ndarray:
        time_error, position_error, normal_alignment, velocity_error = self._tracking_terms()
        cfg = self.cfg.curriculum
        tracking = event_tracking_reward(time_error, position_error, normal_alignment, velocity_error, time_tolerance=cfg.time_tolerance, position_tolerance=cfg.position_tolerance, normal_tolerance=cfg.normal_tolerance, velocity_tolerance=cfg.velocity_tolerance)
        if cfg.stage in (CurriculumStage.REAL_CONTACT, CurriculumStage.SMASH_AND_RECOVER, CurriculumStage.DOMAIN_RANDOMIZED):
            tracking *= 0.25
        attack = np.where(np.array([e.contact for e in self.episode_events]), np.clip(self.attack_quality, 0.0, 1.0), 0.0)
        distance = np.linalg.norm(self.shuttle_pos - self.target_pos, axis=1)
        energy = np.sum(self.joint_vel * self.joint_vel, axis=1)
        risk = np.maximum(0.0, 0.15 - self.base_pos[:, 2]) + 0.05 * np.linalg.norm(self.base_ang_vel, axis=1)
        recovery = np.array([float(e.stable_recovery) for e in self.episode_events])
        success = self.success_pulse.copy()
        return compute_reward(self.contact_pulse, attack, distance, energy, risk, tracking=tracking, recovery=recovery, success=success)

    def get_dones(self) -> tuple[np.ndarray, np.ndarray]:
        terminated = np.array([e.end_reason in (EpisodeEndReason.OUT_OF_BOUNDS, EpisodeEndReason.UNSTABLE, EpisodeEndReason.SUCCESS) for e in self.episode_events], dtype=bool)
        truncated = (self.step_count >= self.cfg.max_steps) & ~terminated
        return terminated, truncated

    def _update_base(self, action: np.ndarray, active: np.ndarray) -> None:
        # A deterministic reduced-order jump model; Isaac Lab replaces it with
        # contacts and articulated dynamics while preserving the event logic.
        legs = action[:, : self.num_leg_joints]
        jump_drive = np.mean(legs[:, :4], axis=1)
        lateral_drive = np.mean(legs[:, 4:6], axis=1)
        # Positive leg thrust must be able to overcome gravity so the staged
        # task has an actual takeoff action to discover.
        self.base_vel[active, 2] += 20.0 * jump_drive[active] * self.cfg.dt - 9.81 * self.cfg.dt
        self.base_vel[active, 1] += 1.5 * lateral_drive[active] * self.cfg.dt
        self.base_vel[active, :2] *= 0.985
        self.base_pos[active] += self.base_vel[active] * self.cfg.dt
        landed = self.base_pos[:, 2] <= self.cfg.nominal_base_height
        self.base_pos[landed, 2] = self.cfg.nominal_base_height
        self.base_vel[landed, 2] = np.maximum(self.base_vel[landed, 2], 0.0)
        self.foot_contacts[:] = landed[:, None]
        self.base_ang_vel[active, 0] = 2.0 * np.mean(legs[active, 6:8], axis=1)
        self.base_ang_vel[active, 1] = 2.0 * np.mean(legs[active, 8:10], axis=1)
        self.base_ang_vel[active, 2] = 2.0 * np.mean(legs[active, 10:12], axis=1)

    def _record_virtual_event(self, index: int) -> None:
        event, metrics = self.episode_events[index], self.episode_metrics[index]
        if event.event_target_reached or self.cfg.curriculum.require_real_contact:
            return
        goal = self.goals[index]
        now = self.step_count[index] * self.cfg.dt
        time_ok = abs(goal.remaining(now)) <= self.cfg.curriculum.time_tolerance
        position_ok = np.linalg.norm(self.racket_pos[index] - goal.position) <= self.cfg.curriculum.position_tolerance
        normal_ok = float(np.dot(self.racket_normal[index], goal.normal)) >= 1.0 - self.cfg.curriculum.normal_tolerance
        velocity_ok = np.linalg.norm(self.racket_vel[index] - goal.velocity) <= self.cfg.curriculum.velocity_tolerance
        if time_ok and position_ok and normal_ok and velocity_ok and event.airborne:
            event.event_target_reached = True
            self.virtual_event_pulse[index] = 1.0
            metrics.contact_count += 1

    def _update_events(self, previous_shuttle: np.ndarray, previous_racket: np.ndarray) -> None:
        self._update_racket_state()
        now = self.step_count.astype(np.float64) * self.cfg.dt
        for i, (event, metrics) in enumerate(zip(self.episode_events, self.episode_metrics)):
            if self.done[i]:
                continue
            metrics.steps = int(self.step_count[i])
            metrics.energy += float(np.sum(self.joint_vel[i] ** 2) * self.cfg.dt)
            metrics.target_error = float(np.linalg.norm(self.shuttle_pos[i] - self.target_pos[i]))
            metrics.max_tilt = max(metrics.max_tilt, float(np.linalg.norm(self.base_ang_vel[i])))
            event.phase = self.phase[i]
            was_airborne = event.airborne
            if not event.takeoff and (self.base_pos[i, 2] > self.cfg.airborne_height or not np.any(self.foot_contacts[i])):
                event.takeoff, event.airborne = True, True
                self.phase[i] = TaskPhase.TAKEOFF
                metrics.takeoff_count += 1
            if event.takeoff and not event.landing and self.base_pos[i, 2] > self.cfg.airborne_height:
                event.airborne = True
                self.phase[i] = TaskPhase.AIRBORNE
            self._record_virtual_event(i)
            if not event.contact and self.cfg.curriculum.require_real_contact:
                racket = RacketGeometry(self.racket_pos[i], self.racket_normal[i], self.cfg.racket_radius)
                hit = swept_sphere_plane_contact(previous_shuttle[i], self.shuttle_pos[i], racket)
                if hit is not None:
                    fraction, point = hit
                    post_velocity, collision = resolve_racket_collision(self.shuttle_vel[i], racket, self.racket_vel[i], time=float(now[i]), contact_position=point)
                    self.shuttle_vel[i] = post_velocity
                    event.contact = True
                    event.airborne_contact = bool(event.airborne or was_airborne)
                    event.contact_time = float(now[i] - self.cfg.dt + fraction * self.cfg.dt)
                    event.contact_position = point.tolist()
                    event.post_contact_velocity = post_velocity.tolist()
                    event.phase = TaskPhase.POST_CONTACT
                    self.phase[i] = TaskPhase.POST_CONTACT
                    self.contact_pulse[i] = 1.0
                    metrics.contact_count += 1
                    metrics.airborne_contact_count += int(event.airborne_contact)
                    self.attack_quality[i] = float(np.clip(max(0.0, post_velocity[0]) / max(1.0, np.linalg.norm(self.goals[i].velocity)), 0.0, 1.0))
            if event.contact and self.shuttle_pos[i, 0] >= self.target_pos[i, 0] and self.shuttle_vel[i, 0] > 0:
                event.valid_return = abs(self.shuttle_pos[i, 1] - self.target_pos[i, 1]) < 1.0 and self.shuttle_pos[i, 2] >= 0.0
                event.attack_requirement = self.attack_quality[i] >= 0.5
                event.end_reason = EpisodeEndReason.RUNNING
                metrics.valid_return_count = int(event.valid_return)
                metrics.attack_quality = float(self.attack_quality[i])
            if event.airborne and not event.landing and self.foot_contacts[i].all() and self.base_pos[i, 2] <= self.cfg.nominal_base_height + 0.012:
                event.landing = True
                event.airborne = False
                event.landing_time = float(now[i])
                event.phase = TaskPhase.RECOVERY
                self.phase[i] = TaskPhase.RECOVERY
            if event.landing and not event.stable_recovery:
                stable = bool(np.linalg.norm(self.base_vel[i]) < 0.25 and np.linalg.norm(self.base_ang_vel[i]) < self.cfg.max_tilt_rate and self.base_pos[i, 2] >= self.cfg.nominal_base_height - 0.01)
                self.recovery_count[i] = self.recovery_count[i] + 1 if stable else 0
                event.recovery_steps = int(self.recovery_count[i])
                if self.recovery_count[i] >= self.cfg.recovery_steps:
                    event.stable_recovery = True
                    metrics.recovery_success_count += 1
            if event.stable_recovery:
                success = outcome_success(np.array([event.airborne_contact]), np.array([event.valid_return]), np.array([event.attack_requirement]), np.array([True]), self.cfg.curriculum, np.array([event.event_target_reached]))[0]
                if success:
                    event.success = True
                    event.end_reason = EpisodeEndReason.SUCCESS
                    event.phase = TaskPhase.COMPLETE
                    self.phase[i] = TaskPhase.COMPLETE
                    self.success_pulse[i] = 1.0
            if self.shuttle_pos[i, 2] < 0.0 or np.linalg.norm(self.shuttle_pos[i, :2]) > 8.0:
                event.out_of_bounds, event.end_reason = True, EpisodeEndReason.OUT_OF_BOUNDS
            if self.base_pos[i, 2] < 0.12 or np.linalg.norm(self.base_ang_vel[i]) > self.cfg.max_tilt_rate * 2.0:
                event.unstable, event.end_reason = True, EpisodeEndReason.UNSTABLE
            if self.step_count[i] >= self.cfg.max_steps and event.end_reason == EpisodeEndReason.RUNNING:
                event.end_reason = EpisodeEndReason.TIMEOUT
            metrics.end_reason = event.end_reason
            event.phase = self.phase[i]

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape == (self.num_joints,):
            action = np.tile(action, (self.num_envs, 1))
        if action.shape != (self.num_envs, self.num_joints) or not np.all(np.isfinite(action)):
            raise ValueError(f"action must have shape ({self.num_envs},{self.num_joints}) and finite values")
        action = np.clip(action, -1.0, 1.0)
        active = ~self.done
        previous_shuttle = self.shuttle_pos.copy()
        previous_racket = self.racket_pos.copy()
        self.contact_pulse.fill(0.0)
        self.success_pulse.fill(0.0)
        self.virtual_event_pulse.fill(0.0)
        scales = np.full(self.num_joints, self.cfg.action_scale, dtype=np.float64)
        scales[self.num_leg_joints :] = self.cfg.arm_action_scale
        desired = action * scales
        self.joint_vel[active] = 0.8 * self.joint_vel[active] + desired[active]
        self.joint_pos[active] = np.clip(self.joint_pos[active] + self.joint_vel[active] * self.cfg.dt, -2.5, 2.5)
        self._update_base(action, active)
        self.shuttle_vel[active] += self.model.acceleration(self.shuttle_vel[active]) * self.cfg.dt
        self.shuttle_pos[active] += self.shuttle_vel[active] * self.cfg.dt
        self.step_count[active] += 1
        self.prev_action[active] = action[active]
        self._update_events(previous_shuttle, previous_racket)
        terminated, truncated = self.get_dones()
        self.done = terminated | truncated
        rewards = self.get_rewards()
        obs = self.get_observations()
        infos = {"episode": [m.as_dict() | {"events": e.as_dict()} for m, e in zip(self.episode_metrics, self.episode_events)], "events": [e.as_dict() for e in self.episode_events], "phase": [p.value for p in self.phase], "curriculum": CurriculumStage(self.cfg.curriculum_stage).value}
        self.last_obs = obs
        return obs.copy(), rewards, terminated, truncated, infos

    def log_episode_metrics(self) -> list[dict]:
        return [metrics.as_dict() | {"events": event.as_dict()} for metrics, event in zip(self.episode_metrics, self.episode_events)]
