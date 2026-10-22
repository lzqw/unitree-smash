"""Analytic racket contact and event recording for the baseline environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _finite3(x: np.ndarray | list[float], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


@dataclass(frozen=True)
class RacketGeometry:
    center: np.ndarray
    normal: np.ndarray
    radius: float = 0.11

    def __post_init__(self) -> None:
        center = _finite3(self.center, "center")
        normal = _finite3(self.normal, "normal")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-10 or self.radius <= 0 or not math.isfinite(self.radius):
            raise ValueError("racket normal must be non-zero and radius must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class CollisionEvent:
    time: float
    contact_position: np.ndarray
    pre_velocity: np.ndarray
    post_velocity: np.ndarray
    normal_impulse: float
    reaction_impulse: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {"time": self.time, "contact_position": self.contact_position.tolist(), "pre_velocity": self.pre_velocity.tolist(), "post_velocity": self.post_velocity.tolist(), "normal_impulse": self.normal_impulse, "reaction_impulse": None if self.reaction_impulse is None else self.reaction_impulse.tolist()}


def swept_sphere_plane_contact(start: np.ndarray, end: np.ndarray, racket: RacketGeometry, sphere_radius: float = 0.033) -> tuple[float, np.ndarray] | None:
    """Return segment fraction and contact point when a moving sphere hits the racket plane."""
    start, end = _finite3(start, "start"), _finite3(end, "end")
    if sphere_radius <= 0 or not math.isfinite(sphere_radius):
        raise ValueError("sphere_radius must be positive")
    d0 = float(np.dot(start - racket.center, racket.normal))
    d1 = float(np.dot(end - racket.center, racket.normal))
    if abs(d0) <= sphere_radius:
        fraction = 0.0
    elif (d0 - sphere_radius) * (d1 - sphere_radius) <= 0 or (d0 + sphere_radius) * (d1 + sphere_radius) <= 0:
        denominator = d0 - d1
        if abs(denominator) < 1e-12:
            return None
        fraction = float(np.clip((d0 - math.copysign(sphere_radius, d0)) / denominator, 0.0, 1.0))
    else:
        return None
    point = start + fraction * (end - start)
    tangent = point - np.dot(point - racket.center, racket.normal) * racket.normal
    if np.linalg.norm(tangent - racket.center) > racket.radius + sphere_radius:
        return None
    return fraction, point


def resolve_racket_collision(pre_velocity: np.ndarray, racket: RacketGeometry, racket_velocity: np.ndarray | None = None, restitution: float = 0.72, friction: float = 0.08, time: float = 0.0, contact_position: np.ndarray | None = None) -> tuple[np.ndarray, CollisionEvent]:
    pre_velocity = _finite3(pre_velocity, "pre_velocity")
    racket_velocity = np.zeros(3) if racket_velocity is None else _finite3(racket_velocity, "racket_velocity")
    if not 0 <= restitution <= 1 or friction < 0 or not math.isfinite(time):
        raise ValueError("restitution must be in [0,1], friction non-negative, time finite")
    relative = pre_velocity - racket_velocity
    normal_speed = float(np.dot(relative, racket.normal))
    if normal_speed >= 0:
        post_velocity = pre_velocity.copy()
        impulse = 0.0
    else:
        normal_component = normal_speed * racket.normal
        tangent_component = relative - normal_component
        post_relative = -restitution * normal_component + max(0.0, 1.0 - friction) * tangent_component
        post_velocity = racket_velocity + post_relative
        impulse = -(1.0 + restitution) * normal_speed
    contact = _finite3(contact_position if contact_position is not None else racket.center, "contact_position")
    return post_velocity, CollisionEvent(time, contact, pre_velocity, post_velocity, impulse)


def resolve_racket_collision_with_reaction(pre_velocity: np.ndarray, racket: RacketGeometry, *, racket_velocity: np.ndarray | None = None, racket_mass: float = math.inf, shuttle_mass: float = 0.0052, restitution: float = 0.72, friction: float = 0.08, time: float = 0.0, contact_position: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, CollisionEvent]:
    """Resolve contact while returning the equal-and-opposite racket impulse.

    ``racket_mass=inf`` reproduces the moving-infinite-mass approximation used
    by the lightweight environment.  Finite mass is available for coupled
    rigid-body validation and never silently changes the configured asset.
    """
    pre_velocity = _finite3(pre_velocity, "pre_velocity")
    racket_velocity = np.zeros(3) if racket_velocity is None else _finite3(racket_velocity, "racket_velocity")
    if not math.isfinite(shuttle_mass) or shuttle_mass <= 0 or racket_mass <= 0 or restitution < 0 or restitution > 1 or friction < 0:
        raise ValueError("masses and collision parameters are invalid")
    relative = pre_velocity - racket_velocity
    normal_speed = float(np.dot(relative, racket.normal))
    if normal_speed >= 0:
        contact = _finite3(contact_position if contact_position is not None else racket.center, "contact_position")
        return pre_velocity.copy(), racket_velocity.copy(), CollisionEvent(time, contact, pre_velocity, pre_velocity.copy(), 0.0, np.zeros(3))
    inv_racket = 0.0 if math.isinf(racket_mass) else 1.0 / racket_mass
    inv_shuttle = 1.0 / shuttle_mass
    impulse_mag = -(1.0 + restitution) * normal_speed / (inv_shuttle + inv_racket)
    impulse = impulse_mag * racket.normal
    post_velocity = pre_velocity + impulse * inv_shuttle
    post_racket_velocity = racket_velocity - impulse * inv_racket
    tangent = relative - normal_speed * racket.normal
    if np.linalg.norm(tangent) > 1e-12:
        tangent_impulse = -min(friction * impulse_mag, np.linalg.norm(tangent) / (inv_shuttle + inv_racket)) * tangent / np.linalg.norm(tangent)
        post_velocity += tangent_impulse * inv_shuttle
        post_racket_velocity -= tangent_impulse * inv_racket
        impulse += tangent_impulse
    contact = _finite3(contact_position if contact_position is not None else racket.center, "contact_position")
    return post_velocity, post_racket_velocity, CollisionEvent(time, contact, pre_velocity, post_velocity, float(impulse_mag), -impulse)


def swept_moving_racket_contact(start: np.ndarray, end: np.ndarray, start_racket: RacketGeometry, end_racket: RacketGeometry, *, sphere_radius: float = 0.033, subdivisions: int = 8) -> tuple[float, np.ndarray, RacketGeometry] | None:
    """Detect a sphere crossing a translating/rotating racket sweep.

    The subdivision is conservative for the dependency-light backend and keeps
    the same valid-face boundary check as :func:`swept_sphere_plane_contact`.
    Isaac Lab can replace it with a continuous collision query while retaining
    this return contract.
    """
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    start, end = _finite3(start, "start"), _finite3(end, "end")
    previous_ball = start
    for index in range(subdivisions):
        a = index / subdivisions
        b = (index + 1) / subdivisions
        center_a = (1.0 - a) * start_racket.center + a * end_racket.center
        center_b = (1.0 - b) * start_racket.center + b * end_racket.center
        normal_a = (1.0 - a) * start_racket.normal + a * end_racket.normal
        normal_b = (1.0 - b) * start_racket.normal + b * end_racket.normal
        racket_a = RacketGeometry(center_a, normal_a, start_racket.radius)
        racket_b = RacketGeometry(center_b, normal_b, start_racket.radius)
        ball_a = start + a * (end - start)
        ball_b = start + b * (end - start)
        hit = swept_sphere_plane_contact(ball_a, ball_b, racket_a, sphere_radius=sphere_radius)
        if hit is not None:
            fraction, point = hit
            local = a + fraction * (b - a)
            normal = (1.0 - fraction) * racket_a.normal + fraction * racket_b.normal
            geometry = RacketGeometry((1.0 - fraction) * racket_a.center + fraction * racket_b.center, normal, racket_a.radius)
            return float(local), point, geometry
        previous_ball = ball_b
    return None

