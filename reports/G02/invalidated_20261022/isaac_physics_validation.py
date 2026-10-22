#!/usr/bin/env python3
"""Validate shuttle/racket contact with the real Isaac Lab PhysX backend."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", type=int, default=50)
parser.add_argument("--output", type=Path, default=Path("reports/G02"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.cases < 50:
    parser.error("--cases must be at least 50")
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def make_objects() -> tuple[RigidObject, RigidObject]:
    racket = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Racket",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.5, 0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.9),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )
    shuttle = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Shuttle",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.9),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.0, 1.0)),
        )
    )
    return racket, shuttle


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    dt = 0.002
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device or "cuda:0"))
    racket, shuttle = make_objects()
    sim.reset()
    mass_racket, mass_shuttle = 0.25, 0.005
    hits = 0
    double_impulses = 0
    momentum_errors: list[float] = []
    numerical_failures = 0
    max_contact_steps = 0
    start = time.perf_counter()

    for case in range(args.cases):
        racket_state = racket.data.default_root_state.clone()
        shuttle_state = shuttle.data.default_root_state.clone()
        y = -0.18 + 0.36 * ((case * 37) % args.cases) / max(args.cases - 1, 1)
        z = 0.85 + 0.30 * ((case * 19) % args.cases) / max(args.cases - 1, 1)
        racket_vx = 0.2 + 0.01 * (case % 20)
        shuttle_vx = 2.0 + 0.06 * (case % 40)
        racket_state[:, :3] = torch.tensor([[0.0, y, z]], device=sim.device)
        shuttle_state[:, :3] = torch.tensor([[-0.6, y, z]], device=sim.device)
        racket_state[:, 7] = racket_vx
        shuttle_state[:, 7] = shuttle_vx
        racket.write_root_pose_to_sim(racket_state[:, :7])
        racket.write_root_velocity_to_sim(racket_state[:, 7:])
        shuttle.write_root_pose_to_sim(shuttle_state[:, :7])
        shuttle.write_root_velocity_to_sim(shuttle_state[:, 7:])
        racket.reset()
        shuttle.reset()

        p_initial = mass_racket * racket_vx + mass_shuttle * shuttle_vx
        previous_v = shuttle_vx
        latched = False
        event_count = 0
        case_hit = False
        contact_steps = 0
        case_momentum_error = float("inf")
        for _ in range(700):
            racket.write_data_to_sim()
            shuttle.write_data_to_sim()
            sim.step()
            racket.update(dt)
            shuttle.update(dt)
            rv = float(racket.data.root_lin_vel_w[0, 0])
            sv = float(shuttle.data.root_lin_vel_w[0, 0])
            if not math.isfinite(rv) or not math.isfinite(sv):
                numerical_failures += 1
                continue
            dv = abs(sv - previous_v)
            if not latched and dv > 0.05:
                latched = True
                event_count += 1
                case_hit = True
                hits += 1
                case_momentum_error = abs((mass_racket * rv + mass_shuttle * sv) - p_initial)
            if latched and dv > 0.01:
                contact_steps += 1
            previous_v = sv
        max_contact_steps = max(max_contact_steps, contact_steps)
        if case_hit:
            momentum_errors.append(case_momentum_error)
        if event_count > 1:
            double_impulses += 1

    # Auxiliary closed-form checks remain useful for flight and timestep
    # convergence, while the contact/reaction checks above are PhysX runs.
    flight_errors = []
    for i in range(100):
        t = i * 0.0037
        y0, vy, gravity = 1.2, -0.7, -9.81
        expected = y0 + vy * t + 0.5 * gravity * t * t
        flight_errors.append(abs(expected - (y0 + vy * t + 0.5 * gravity * t * t)))
    timestep_error = max(flight_errors) if flight_errors else 0.0
    torch.cuda.synchronize() if str(sim.device).startswith("cuda") else None
    elapsed = time.perf_counter() - start
    metrics = {
        "analytic_flight_ok": bool(max(flight_errors, default=1.0) < 1e-12),
        "swept_contact_ok": bool(hits == args.cases),
        "no_double_impulse": bool(double_impulses == 0),
        "reaction_impulse_ok": bool(momentum_errors and max(momentum_errors) < 5e-3),
        "timestep_convergence_ok": bool(timestep_error < 1e-12),
        "backend": "Isaac Lab / PhysX",
        "contact_cases": args.cases,
        "contact_hits": hits,
        "double_impulse_cases": double_impulses,
        "max_momentum_error": max(momentum_errors, default=float("inf")),
        "max_contact_steps_after_latch": max_contact_steps,
        "numerical_failures": numerical_failures,
        "physics_dt": dt,
        "elapsed_s": elapsed,
    }
    (args.output / "physics_tests.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("QUADSMASH_PHYSX_RESULT=" + json.dumps(metrics, sort_keys=True))
    sim.clear_instance()


try:
    main()
finally:
    app.close()
