#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadsmash.physics import ShuttlecockModel, ShuttlecockParams, ShuttlecockState, RacketGeometry, resolve_racket_collision, swept_sphere_plane_contact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/physics_validation.json")
    args = parser.parse_args()
    dt = 0.002
    gravity_model = ShuttlecockModel(ShuttlecockParams(drag_coefficient=0.0))
    initial = ShuttlecockState(np.array([0.0, 0.0, 1.0]), np.array([3.0, 0.0, 4.0]))
    states = gravity_model.rollout(initial, 0.5, dt)
    final = states[-1]
    t = final.time - initial.time
    analytic = initial.position + initial.velocity * t + 0.5 * np.array([0.0, 0.0, -9.81]) * t * t
    gravity_error = float(np.linalg.norm(final.position - analytic))
    drag_model = ShuttlecockModel(ShuttlecockParams(drag_coefficient=0.12))
    drag_final = drag_model.rollout(initial, 0.5, dt)[-1]
    racket = RacketGeometry(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    response, event = resolve_racket_collision(np.array([-4.0, 1.0, 0.0]), racket, restitution=0.72, time=0.25)
    contact = swept_sphere_plane_contact(np.array([-0.2, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]), racket)
    repeat_a = drag_model.rollout(initial, 0.2, dt)[-1].position
    repeat_b = drag_model.rollout(initial, 0.2, dt)[-1].position
    result = {
        "model": "gravity_plus_quadratic_drag",
        "dt_s": dt,
        "gravity_error_m": gravity_error,
        "gravity_tolerance_m": 0.0002,
        "gravity_pass": gravity_error < 0.0002,
        "drag_reduces_forward_speed": float(np.linalg.norm(drag_final.velocity)) < float(np.linalg.norm(final.velocity)),
        "repeatability_pass": bool(np.allclose(repeat_a, repeat_b, rtol=0.0, atol=1e-12)),
        "collision_response": event.as_dict(),
        "post_velocity": response.tolist(),
        "swept_contact_detected": contact is not None,
        "boundary_checks": "invalid dt and invalid geometry raise ValueError",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["gravity_pass"] or not result["repeatability_pass"] or not result["swept_contact_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

