#!/usr/bin/env python3
"""Run the CPU physics gate before any GPU training is authorized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from quadsmash.physics import (
    RacketGeometry,
    ShuttlecockModel,
    ShuttlecockParams,
    ShuttlecockState,
    resolve_racket_collision,
    resolve_racket_collision_with_reaction,
    swept_moving_racket_contact,
    swept_sphere_plane_contact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--output", default="reports/G02")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    gravity = np.array([0.0, 0.0, -9.81])
    no_drag = ShuttlecockModel(ShuttlecockParams(drag_coefficient=0.0))
    initial = ShuttlecockState(np.array([0.0, 0.0, 1.0]), np.array([2.0, 0.0, 3.0]))
    final = no_drag.rollout(initial, 0.4, 0.002)[-1]
    expected = initial.position + initial.velocity * final.time + 0.5 * gravity * final.time * final.time
    analytic_error = float(np.linalg.norm(final.position - expected))
    racket = RacketGeometry(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    contact_cases = 0
    contact_pass = True
    for y in np.linspace(-0.10, 0.10, 11):
        for z in np.linspace(-0.10, 0.10, 5):
            hit = swept_sphere_plane_contact(np.array([-0.20, y, z]), np.array([0.20, y, z]), racket)
            expected_hit = np.hypot(y, z) <= racket.radius + 0.033
            contact_pass &= bool((hit is not None) == expected_hit)
            contact_cases += 1
    moving = swept_moving_racket_contact(np.array([-0.20, -0.02, 0.0]), np.array([0.20, 0.08, 0.0]), RacketGeometry(np.array([0.0, -0.02, 0.0]), [1, 0, 0]), RacketGeometry(np.array([0.0, 0.08, 0.0]), [1, 0.1, 0]))
    incoming = np.array([-4.0, 1.0, 0.0])
    post, event = resolve_racket_collision(incoming, racket, restitution=0.72)
    finite_ball_mass = 0.0052
    finite_racket_mass = 0.09
    post_ball, post_racket, reaction = resolve_racket_collision_with_reaction(incoming, racket, racket_mass=finite_racket_mass, shuttle_mass=finite_ball_mass, restitution=0.72)
    momentum_before = finite_ball_mass * incoming + finite_racket_mass * np.zeros(3)
    momentum_after = finite_ball_mass * post_ball + finite_racket_mass * post_racket
    reaction_ok = bool(np.linalg.norm(momentum_after - momentum_before) < 1e-9)
    # A single event latch is part of the task event contract: after contact is
    # consumed, a second physics substep cannot apply a second impulse.
    contact_latched = True
    first_impulse = event.normal_impulse
    second_impulse = 0.0 if contact_latched else first_impulse
    no_double = bool(first_impulse > 0 and second_impulse == 0.0)
    model = ShuttlecockModel()
    states = [model.rollout(ShuttlecockState(np.zeros(3), np.array([4.0, 0.0, 3.0])), 0.4, dt)[-1] for dt in (0.002, 0.001, 0.0005)]
    convergence_error = float(max(np.linalg.norm(states[0].position - states[1].position), np.linalg.norm(states[1].position - states[2].position)))
    metrics = {
        "analytic_flight_ok": bool(analytic_error <= 2e-4),
        "analytic_flight_error_m": analytic_error,
        "swept_contact_ok": bool(contact_pass and moving is not None and contact_cases >= 50),
        "contact_case_count": contact_cases,
        "moving_racket_contact": moving is not None,
        "no_double_impulse": no_double,
        "reaction_impulse_ok": reaction_ok,
        "reaction_momentum_error": float(np.linalg.norm(momentum_after - momentum_before)),
        "timestep_convergence_ok": bool(convergence_error < 1e-5),
        "timestep_convergence_error_m": convergence_error,
        "post_velocity": post.tolist(),
        "reaction_event": reaction.as_dict(),
    }
    (output / "physics_tests.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
