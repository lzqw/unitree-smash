#!/usr/bin/env python3
"""Measure native floating-base reset, partial isolation and zero-action physics."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--asset", type=Path)
parser.add_argument("--root-height", type=float, default=0.4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
started = time.time()
launcher = AppLauncher(args)
app = launcher.app

import torch
from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg

cfg = JumpEnvCfg()
cfg.seed = 1907
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device or "cuda:0"
cfg.robot.init_state.pos = (0.0, 0.0, args.root_height)
if args.asset:
    cfg.robot.spawn.usd_path = str(args.asset.resolve())
env = JumpEnv(cfg)
try:
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    colliders = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (str(prim.GetPath()).startswith("/World/envs/env_0/") or str(prim.GetPath()).startswith("/World/Ground")):
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
            colliders.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                              "body": str(body.GetPath()) if body else "STATIC",
                              "min": list(box.GetMin()), "max": list(box.GetMax())})
    (args.output / "colliders.json").write_text(json.dumps(colliders, indent=2) + "\n")
    obs, _ = env.reset()
    expected = env.robot.data.default_root_state.clone()
    expected[:, :3] += env.scene.env_origins
    initial_error = (env.robot.data.root_state_w - expected).abs().max().item()
    mass = env.robot.root_physx_view.get_masses().to(env.device)
    feet, names = env.robot.find_bodies(".*_foot")
    trace = []
    with torch.inference_mode():
        for step in range(args.steps):
            obs, rew, term, trunc, _ = env.step(torch.zeros((args.num_envs, 18), device=env.device))
            if step % 10 == 0 or bool(term.any()):
                data = env.robot.data
                com = (data.body_com_pos_w * mass.unsqueeze(-1)).sum(1) / mass.sum(1, keepdim=True)
                trace.append({"step": step + 1, "base_z": data.root_pos_w[0, 2].item(),
                              "com_z": com[0, 2].item(), "vz": data.root_lin_vel_w[0, 2].item(),
                              "gravity_z": data.projected_gravity_b[0, 2].item(),
                              "feet_z_min": data.body_pos_w[0, feet, 2].min().item(),
                              "feet_force_max": env.feet_contact.data.net_forces_w[0].norm(dim=-1).max().item(),
                              "joint_speed_max": data.joint_vel[0].abs().max().item(),
                              "terminated": bool(term[0]), "truncated": bool(trunc[0]),
                              "episode_step": int(env.episode_length_buf[0])})
    before_other = env.robot.data.root_state_w[1:].clone()
    before_q_other = env.robot.data.joint_pos[1:].clone()
    # Deliberately leave inference mode: training followed by reset must work.
    env._reset_idx(torch.tensor([0], device=env.device))
    result = {
        "backend": "Isaac Lab / PhysX", "fixed_base": env.robot.is_fixed_base,
        "mass_kg": float(mass[0].sum()), "body_names": env.robot.body_names,
        "asset": cfg.robot.spawn.usd_path, "root_height": args.root_height,
        "body_masses_kg": mass[0].tolist(), "feet": names,
        "initial_root_state_max_error": initial_error,
        "reset_root_state_max_error": (env.robot.data.root_state_w[0] - expected[0]).abs().max().item(),
        "reset_joint_position_max_error": (env.robot.data.joint_pos[0] - env.robot.data.default_joint_pos[0]).abs().max().item(),
        "reset_joint_velocity_max_error": env.robot.data.joint_vel[0].abs().max().item(),
        "other_root_state_max_change": (env.robot.data.root_state_w[1:] - before_other).abs().max().item(),
        "other_joint_position_max_change": (env.robot.data.joint_pos[1:] - before_q_other).abs().max().item(),
        "num_envs": args.num_envs, "steps": args.steps,
        "elapsed_s": time.time() - started, "command": sys.argv,
        "episodes": env.episode_records,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "trace.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    print("RESET_AUDIT=" + json.dumps({k: v for k, v in result.items() if k not in ("episodes", "body_names", "body_masses_kg", "command")}))
except BaseException:
    import traceback
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error)
    print(error, flush=True)
    raise
finally:
    env.close()
    app.close()
