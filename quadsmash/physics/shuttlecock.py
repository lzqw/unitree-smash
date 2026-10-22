"""Small deterministic shuttlecock model used by the numerical baseline.

The model is deliberately explicit: it includes gravity and quadratic drag and
keeps orientation/angular velocity in the state so an Isaac Lab rigid body can
replace this integrator without changing task interfaces.  It is a calibration
model, not a claim that the placeholder parameters represent a certified ball.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


def _as3(value: np.ndarray | list[float] | tuple[float, float, float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("orientation must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return q / norm


@dataclass(frozen=True)
class ShuttlecockParams:
    mass: float = 0.0052
    drag_coefficient: float = 0.12
    air_density: float = 1.225
    reference_area: float = 0.0018
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_speed: float = 200.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.mass, self.drag_coefficient, self.air_density, self.reference_area, self.max_speed)):
            raise ValueError("physical parameters must be finite")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if self.mass <= 0 or self.drag_coefficient < 0 or self.air_density < 0 or self.reference_area < 0:
            raise ValueError("mass must be positive and aerodynamic parameters non-negative")
        _as3(self.gravity, "gravity")
        _as3(self.wind, "wind")

    @property
    def gravity_array(self) -> np.ndarray:
        return np.asarray(self.gravity, dtype=np.float64)


@dataclass
class ShuttlecockState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0

    def __post_init__(self) -> None:
        self.position = _as3(self.position, "position")
        self.velocity = _as3(self.velocity, "velocity")
        self.orientation = _normalize_quaternion(np.asarray(self.orientation, dtype=np.float64))
        self.angular_velocity = _as3(self.angular_velocity, "angular_velocity")
        if self.time < 0 or not math.isfinite(self.time):
            raise ValueError("time must be finite and non-negative")

    def copy(self) -> "ShuttlecockState":
        return ShuttlecockState(self.position.copy(), self.velocity.copy(), self.orientation.copy(), self.angular_velocity.copy(), self.time)


class ShuttlecockModel:
    """RK4 translation integrator with quadratic aerodynamic drag.

    The v1 model intentionally keeps orientation and angular velocity in the
    state contract but does not integrate rotational dynamics. Non-zero spin is
    rejected so the simplified model cannot silently claim to model it.
    """

    def __init__(self, params: ShuttlecockParams | None = None):
        self.params = params or ShuttlecockParams()

    def acceleration(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim == 0 or velocity.shape[-1] != 3 or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must have a final dimension of three")
        relative = velocity - np.asarray(self.params.wind, dtype=np.float64)
        speed = np.linalg.norm(relative, axis=-1, keepdims=True)
        if np.any(np.linalg.norm(velocity, axis=-1) > self.params.max_speed):
            raise ValueError("velocity exceeds configured model range")
        drag = np.zeros_like(velocity, dtype=np.float64)
        if self.params.drag_coefficient > 0 and self.params.reference_area > 0:
            scale = 0.5 * self.params.air_density * self.params.drag_coefficient * self.params.reference_area / self.params.mass
            drag = -scale * speed * relative
        return self.params.gravity_array + drag

    def step(self, state: ShuttlecockState, dt: float) -> ShuttlecockState:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if np.any(state.angular_velocity != 0):
            raise ValueError("v1 does not integrate rotational dynamics")
        next_position, next_velocity = self.integrate(state.position, state.velocity, dt)
        return ShuttlecockState(next_position, next_velocity, state.orientation.copy(), state.angular_velocity.copy(), state.time + dt)

    def integrate(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Integrate one translation step for arrays shaped ``[..., 3]``."""
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if position.shape != velocity.shape or position.ndim == 0 or position.shape[-1] != 3:
            raise ValueError("position and velocity must have the same final dimension of three")
        if not np.all(np.isfinite(position)) or not math.isfinite(dt) or not 0 < dt <= 0.01:
            raise ValueError("states must be finite and 0 < dt <= 0.01")
        k1_v = self.acceleration(velocity)
        k2_v = self.acceleration(velocity + 0.5 * dt * k1_v)
        k3_v = self.acceleration(velocity + 0.5 * dt * k2_v)
        k4_v = self.acceleration(velocity + dt * k3_v)
        next_velocity = velocity + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
        next_position = position + dt * (velocity + 2 * (velocity + 0.5 * dt * k1_v) + 2 * (velocity + 0.5 * dt * k2_v) + (velocity + dt * k3_v)) / 6.0
        return next_position, next_velocity

    def rollout(self, initial: ShuttlecockState, duration: float, dt: float) -> list[ShuttlecockState]:
        if duration < 0 or not math.isfinite(duration) or not math.isfinite(dt) or not 0 < dt <= 0.01:
            raise ValueError("duration must be finite and non-negative")
        states = [initial.copy()]
        state = initial.copy()
        count = int(math.ceil(duration / dt)) if duration else 0
        for index in range(count):
            state = self.step(state, min(dt, duration - index * dt))
            states.append(state)
        return states

