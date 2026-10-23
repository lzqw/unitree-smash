"""CPU free-flight necessary conditions under the locked drag/gravity model.

Predicted outgoing velocities are search targets. This module never applies
them to the simulator and cannot certify a robot or racket collision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def predict_flights(position, velocity, *, dt=.002, duration=3., drag=None,
                    net_x=1.5, net_height=1.524, radius=.025,
                    net_half_width=1., net_half_thickness=.01,
                    target_x=(2., 4.), target_y=1.):
    """Batched RK4 prediction, ending at the first ground crossing (no bounce)."""
    position, velocity = np.atleast_2d(position).astype(float), np.atleast_2d(velocity).astype(float)
    position, velocity = np.broadcast_arrays(position, velocity)
    if position.shape[-1] != 3 or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("Expected finite position/velocity arrays with final dimension 3")
    if not 0 < dt <= .01 or duration <= 0:
        raise ValueError("Invalid flight integration interval")
    config = json.loads((ROOT / "configs/shuttle_physics_v2.json").read_text())
    drag = config["quadratic_drag_k_per_m"] if drag is None else drag
    wind = np.asarray(config["wind_m_s"])
    initial_position, initial_velocity = position.copy(), velocity.copy()
    n = len(position)
    net_position, landing_position = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    net_time, landing_time = np.full(n, np.nan), np.full(n, np.nan)
    exit_time, exit_margin = np.full(n, np.nan), np.full(n, np.nan)
    active = position[:, 2] > radius
    def acceleration(v):
        relative = v - wind
        return np.array([0., 0., -9.81]) - drag * np.linalg.norm(relative, axis=1, keepdims=True) * relative
    for tick in range(int(np.ceil(duration / dt))):
        if not active.any():
            break
        interval = min(dt, duration - tick * dt)
        old = position.copy()
        k1 = acceleration(velocity)
        v2 = velocity + .5 * interval * k1
        k2 = acceleration(v2)
        v3 = velocity + .5 * interval * k2
        k3 = acceleration(v3)
        v4 = velocity + interval * k3
        k4 = acceleration(v4)
        position = old + interval * (velocity + 2*v2 + 2*v3 + v4) / 6
        velocity = velocity + interval * (k1 + 2*k2 + 2*k3 + k4) / 6
        crossed = active & np.isnan(net_time) & (old[:, 0] < net_x) & (position[:, 0] >= net_x)
        fraction = np.clip((net_x - old[:, 0]) / np.where(position[:, 0] != old[:, 0], position[:, 0] - old[:, 0], 1), 0, 1)
        point = old + fraction[:, None] * (position - old)
        net_position[crossed] = point[crossed]
        net_time[crossed] = tick * dt + fraction[crossed] * interval
        # For a downward shot, clearance at the rear of the finite net is
        # stricter than its center plane. The expanded X slab below is a
        # conservative sufficient condition for clearing the actual box.
        exit_x = net_x + net_half_thickness + radius
        exited = active & np.isnan(exit_time) & (old[:, 0] < exit_x) & (position[:, 0] >= exit_x)
        fraction = np.clip((exit_x - old[:, 0]) / np.where(position[:, 0] != old[:, 0], position[:, 0] - old[:, 0], 1), 0, 1)
        point = old + fraction[:, None] * (position - old)
        exit_margin[exited] = point[exited, 2] - radius - net_height
        exit_time[exited] = tick * dt + fraction[exited] * interval
        landed = active & (old[:, 2] > radius) & (position[:, 2] <= radius)
        fraction = np.clip((radius - old[:, 2]) / np.where(position[:, 2] != old[:, 2], position[:, 2] - old[:, 2], 1), 0, 1)
        point = old + fraction[:, None] * (position - old)
        landing_position[landed] = point[landed]
        landing_time[landed] = tick * dt + fraction[landed] * interval
        active &= ~landed
    margin = net_position[:, 2] - radius - net_height
    net_clear = ((margin > 0) & (np.abs(net_position[:, 1]) + radius <= net_half_width)
                 & (net_time < landing_time) & (exit_margin > 0) & (exit_time < landing_time))
    target = ((landing_position[:, 0] >= target_x[0]) & (landing_position[:, 0] <= target_x[1])
              & (np.abs(landing_position[:, 1]) <= target_y))
    attack = ((np.linalg.norm(initial_velocity, axis=1) >= 8) & (initial_velocity[:, 0] > 0)
              & (initial_velocity[:, 2] <= -1))
    return dict(position=initial_position, velocity=initial_velocity, net_position=net_position,
                landing_position=landing_position, net_time=net_time, landing_time=landing_time,
                net_margin=margin, conservative_net_exit_margin=exit_margin,
                predicted_legal_attack=attack & net_clear & target,
                status="free-flight necessary condition only; no robot collision validation")
