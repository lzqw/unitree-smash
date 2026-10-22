#!/usr/bin/env python3
"""Independent standing and free-flight reaction audit for the corrected asset."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--self-collisions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from quadsmash.tasks.stand.coupled_env import CoupledEnv, CoupledEnvCfg, EXPECTED_JOINTS

cfg = CoupledEnvCfg()
cfg.seed = 1907
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 11.0
cfg.robot.spawn.usd_path = str(args.asset.resolve())
cfg.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions
cfg.robot.init_state.pos = (0.0, 0.0, 0.323)
cfg.sim.device = args.device or "cuda:0"
env = CoupledEnv(cfg)


def momentum():
    data = env.robot.data
    mass = env.robot.root_physx_view.get_masses().to(env.device)
    inertia = env.robot.root_physx_view.get_inertias().to(env.device).reshape(env.num_envs, -1, 3, 3)
    positions = data.body_com_pos_w
    velocity = data.body_com_lin_vel_w
    com = (positions * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
    vcom = (velocity * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
    # get_inertias returns the full tensor in each body/link frame. Do not
    # rotate it with the principal-axis COM quaternion a second time.
    omega_local = quat_apply_inverse(data.body_quat_w, data.body_ang_vel_w)
    own = quat_apply(data.body_quat_w, (inertia @ omega_local[..., None]).squeeze(-1))
    angular = (torch.linalg.cross(positions - com[:, None], mass[..., None] * (velocity - vcom[:, None]), dim=-1) + own).sum(1)
    return com, vcom, angular


try:
    rows = []
    numerical = 0
    with torch.inference_mode():
        for batch in range(math.ceil(args.episodes / args.num_envs)):
            env.reset()
            min_height = torch.full((env.num_envs,), float("inf"), device=env.device)
            max_tilt = torch.zeros(env.num_envs, device=env.device)
            max_torque = torch.zeros(env.num_envs, device=env.device)
            max_speed = torch.zeros(env.num_envs, device=env.device)
            premature_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for step in range(1000):
                actions = torch.zeros((env.num_envs, 18), device=env.device)
                actions[:, 12:] = 0.5 * math.sin(2 * math.pi * step * env.step_dt / 5)
                obs, reward, terminated, truncated, _ = env.step(actions)
                data = env.robot.data
                numerical += int((~torch.isfinite(obs["policy"])).sum())
                min_height = torch.minimum(min_height, data.root_pos_w[:, 2])
                tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
                max_tilt = torch.maximum(max_tilt, tilt)
                max_torque = torch.maximum(max_torque, data.applied_torque.abs().max(1).values)
                max_speed = torch.maximum(max_speed, data.joint_vel.abs().max(1).values)
                premature_reset |= terminated | truncated
            for i in range(min(env.num_envs, args.episodes - len(rows))):
                rows.append({"episode": len(rows), "duration_s": 10.0,
                             "minimum_base_height_m": min_height[i].item(),
                             "maximum_tilt_deg": math.degrees(max_tilt[i].item()),
                             "maximum_joint_torque_nm": max_torque[i].item(),
                             "maximum_joint_speed_rad_s": max_speed[i].item(),
                             "premature_reset": bool(premature_reset[i]),
                             "stand_success": bool(min_height[i] > 0.15 and max_tilt[i] < math.radians(30) and not premature_reset[i])})
        # Diagnostic initialization only: a paired free-flight experiment tests
        # reaction/momentum without ground forces. It is never counted in G04.
        env.robot.data.default_root_state[:, 2] = 5.0
        env.reset()
        env.step(torch.zeros((env.num_envs, 18), device=env.device))
        com0, vel0, angular0 = [v.clone() for v in momentum()]
        com_error, angular_drift, reaction = 0.0, 0.0, 0.0
        for step in range(50):
            actions = torch.zeros((env.num_envs, 18), device=env.device)
            actions[1::2, 12:] = math.sin(2 * math.pi * step / 50)
            env.step(actions)
            com, vel, angular = momentum()
            t = (step + 1) * env.step_dt
            expected = com0 + vel0 * t + torch.tensor([0, 0, -4.905*t*t], device=env.device)
            com_error = max(com_error, (com - expected).norm(dim=-1).max().item())
            angular_drift = max(angular_drift, (angular - angular0).norm(dim=-1).max().item())
            difference = env.robot.data.root_ang_vel_w[1::2] - env.robot.data.root_ang_vel_w[0::2]
            reaction = max(reaction, difference.norm(dim=-1).max().item())
    mass = env.robot.root_physx_view.get_masses()[0].tolist()
    result = {"backend": "Isaac Lab / PhysX", "asset": str(args.asset.resolve()),
              "self_collisions_enabled": args.self_collisions,
              "fixed_base": bool(env.robot.is_fixed_base), "mass_kg": sum(mass),
              "joint_mapping_ok": tuple(env.robot.joint_names[i] for i in env.joint_ids) == EXPECTED_JOINTS,
              "sample_count": len(rows), "stand_episode_seconds": 10.0,
              "stand_success_rate": sum(r["stand_success"] for r in rows) / len(rows) if rows else None,
              "numerical_failures": numerical,
              "free_flight_com_error_m": com_error,
              "free_flight_angular_momentum_drift_kg_m2_s": angular_drift,
              "paired_arm_motion_base_reaction_rad_s": reaction,
              "reaction_detected": reaction > 0.001,
              "mass_by_body": dict(zip(env.robot.body_names, mass)),
              "note": "Partial G01 re-audit only. Does not complete reachability/contact calibration or G04 training."}
    if rows:
        with (args.output / "episodes.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ASSET_REAUDIT=" + json.dumps(result), flush=True)
except BaseException:
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    env.close()
    app.close()
