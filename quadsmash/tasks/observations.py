"""Observation construction kept separate from privileged simulator state."""

from __future__ import annotations

import numpy as np

from .goals import HitEventGoal, TaskPhase, phase_one_hot


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions with shape (...,4) to rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternion last dimension must be four")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = [q[..., i] for i in range(4)]
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    base_pos: np.ndarray,
    base_vel: np.ndarray,
    base_quat: np.ndarray,
    shuttle_pos: np.ndarray,
    shuttle_vel: np.ndarray,
    shuttle_quat: np.ndarray,
    shuttle_omega: np.ndarray,
    target_pos: np.ndarray,
    foot_contacts: np.ndarray,
    *,
    now: np.ndarray | None = None,
    goals: tuple[HitEventGoal, ...] | list[HitEventGoal] | None = None,
    racket_pos: np.ndarray | None = None,
    racket_vel: np.ndarray | None = None,
    racket_normal: np.ndarray | None = None,
    phases: list[TaskPhase] | tuple[TaskPhase, ...] | None = None,
    event_flags: np.ndarray | None = None,
    previous_actions: np.ndarray | None = None,
    base_ang_vel: np.ndarray | None = None,
) -> np.ndarray:
    """Build policy observations, including the time-conditioned hit event.

    The original P2.5 state remains the prefix of the vector.  Optional task
    fields are enabled by the environment and are intentionally explicit so a
    deployment adapter can expose the same contract.  ``event_flags`` has
    columns ``takeoff, contact, landing`` and ``previous_actions`` is the last
    normalized 18-D action.
    """
    base_rot = quaternion_to_matrix(base_quat).reshape(base_quat.shape[0], -1)
    if base_ang_vel is None:
        base_ang_vel = np.zeros_like(base_vel)
    # Keep the original P2.5 71-element vector as an exact prefix.
    parts = [joint_pos, joint_vel, base_pos, base_vel, base_rot, shuttle_pos, shuttle_vel, shuttle_quat, shuttle_omega, target_pos, foot_contacts.astype(np.float64)]
    n = joint_pos.shape[0]
    if goals is None:
        goals = tuple(HitEventGoal(0.0, target_pos[i], np.array([1.0, 0.0, 0.0]), np.zeros(3)) for i in range(n))
    if now is None:
        now = np.zeros(n, dtype=np.float64)
    goal_vectors = np.stack([goals[i].as_vector(float(now[i])) for i in range(n)], axis=0)
    parts.append(goal_vectors)

    if racket_pos is None:
        racket_pos = np.zeros((n, 3), dtype=np.float64)
    if racket_vel is None:
        racket_vel = np.zeros((n, 3), dtype=np.float64)
    if racket_normal is None:
        racket_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    parts.extend((racket_pos, racket_vel, racket_normal))

    if phases is None:
        phases = [TaskPhase.PREPARE] * n
    parts.append(np.stack([phase_one_hot(phase) for phase in phases], axis=0))
    if event_flags is None:
        event_flags = np.zeros((n, 3), dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=np.float64)
    if event_flags.shape != (n, 3):
        raise ValueError("event_flags must have shape (num_envs, 3)")
    parts.append(event_flags)
    if previous_actions is None:
        previous_actions = np.zeros((n, joint_pos.shape[1]), dtype=np.float64)
    parts.extend((np.asarray(previous_actions, dtype=np.float64), np.asarray(base_ang_vel, dtype=np.float64)))
    return np.concatenate(parts, axis=1).astype(np.float32)


def observation_size(num_joints: int = 18) -> int:
    """Return the event-conditioned observation width.

    The 71-element P2.5 state is followed by goal (10), racket state (9),
    phase (5), event flags (3), and the previous action (``num_joints``).
    """
    return 2 * num_joints + 3 + 3 + 3 + 9 + 3 + 3 + 4 + 3 + 3 + 4 + 10 + 9 + 5 + 3 + num_joints

