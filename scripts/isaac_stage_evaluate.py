#!/usr/bin/env python3
"""Frozen ground-initialized evaluator for native QuadSmash stages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("jump", "virtual_hit", "real_contact", "jump_smash"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--config", type=Path, default=Path("configs/jump_ppo.json"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=1907)
parser.add_argument("--suite", default=None)
parser.add_argument("--reset-mode", choices=("ground",), default="ground")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.num_envs < 1:
    parser.error("episodes and num-envs must be positive")
args.output.mkdir(parents=True, exist_ok=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from quadsmash.tasks.stand.ball_env import BallEnv, BallEnvCfg
from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg
from quadsmash.tasks.stand.virtual_env import VirtualHitEnv, VirtualHitEnvCfg


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def choose_env():
    if args.stage == "jump":
        return JumpEnvCfg(), JumpEnv, "jump_recover_ground_v1"
    if args.stage == "virtual_hit":
        return VirtualHitEnvCfg(), VirtualHitEnv, "virtual_hit_recover_ground_v1"
    return BallEnvCfg(), BallEnv, (
        "airborne_contact_recover_ground_v1" if args.stage == "real_contact" else "airborne_smash_recover_net1524_v1"
    )


def main() -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg, env_type, task_name = choose_env()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = config.get("episode_seconds", 4.0)
    if args.stage == "jump_smash":
        cfg.include_net = True
    env = env_type(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, config["runner"], log_dir=str(args.output / "ppo_eval"), device=env.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    started = time.perf_counter()
    try:
        obs, _ = wrapped.reset()
        initial_height = env.robot.data.root_pos_w[:, 2].detach().clone()
        ground_init = bool(torch.isfinite(initial_height).all() and ((initial_height > 0.35) & (initial_height < 0.50)).all())
        numerical_failures = int(not bool(torch.isfinite(obs["policy"]).all().item()))
        with torch.inference_mode():
            while len(getattr(env, "episode_records", [])) < args.episodes:
                action = policy(obs)
                numerical_failures += int(not bool(torch.isfinite(action).all().item()))
                obs, reward, _, _ = wrapped.step(action)
                numerical_failures += int(not bool(torch.isfinite(obs["policy"]).all().item()))
                numerical_failures += int(not bool(torch.isfinite(reward).all().item()))

        if args.stage == "virtual_hit":
            raw = env.virtual_episode_records[: args.episodes]
            rows = []
            for index, item in enumerate(raw):
                rows.append({"episode": index, "ground_init": ground_init, **item})
            success = [bool(row["virtual_event"] and row["recovery_success"]) for row in rows]
            metrics = {
                "sample_count": len(rows),
                "ground_init_fraction": sum(bool(row["ground_init"]) for row in rows) / len(rows),
                "virtual_hit_recovery_success_rate": sum(success) / len(rows),
                "numerical_failures": numerical_failures,
            }
        elif args.stage in ("real_contact", "jump_smash"):
            raw = env.ball_episode_records[: args.episodes]
            rows = []
            for index, item in enumerate(raw):
                rows.append({"episode": index, "ground_init": ground_init, **item})
            contact = [bool(row["contact"]) for row in rows]
            recovered = [bool(row["recovery_success"]) for row in rows]
            contact_recovery = [c and r for c, r in zip(contact, recovered)]
            if args.stage == "real_contact":
                metrics = {
                    "sample_count": len(rows),
                    "ground_init_fraction": sum(bool(row["ground_init"]) for row in rows) / len(rows),
                    "airborne_contact_recovery_success_rate": sum(contact_recovery) / len(rows),
                    "numerical_failures": numerical_failures,
                }
            else:
                attack = [
                    c and row["outgoing_speed_mps"] >= 8.0 and row["outgoing_vz_mps"] <= -1.0
                    for c, row in zip(contact, rows)
                ]
                complete = [
                    a and row["crossed_net"] and row["landed_target"] and row["recovery_success"]
                    for a, row in zip(attack, rows)
                ]
                metrics = {
                    "sample_count": len(rows),
                    "ground_init_fraction": sum(bool(row["ground_init"]) for row in rows) / len(rows),
                    "complete_smash_success_rate": sum(complete) / len(rows),
                    "airborne_contact_rate": sum(contact) / len(rows),
                    "recovery_given_airborne_contact": sum(r for c, r in zip(contact, recovered) if c) / max(sum(contact), 1),
                    "numerical_failures": numerical_failures,
                }
        else:
            raw = env.episode_records[: args.episodes]
            rows = []
            for index, item in enumerate(raw):
                rows.append({"episode": index, "ground_init": ground_init, **item})
            success = [bool(row["success"] and row["ground_init"]) for row in rows]
            metrics = {
                "sample_count": len(rows),
                "ground_init_fraction": sum(bool(row["ground_init"]) for row in rows) / len(rows),
                "jump_recovery_success_rate": sum(success) / len(rows),
                "numerical_failures": numerical_failures,
            }
        elapsed = time.perf_counter() - started
        metrics.update({"backend": "Isaac Lab / PhysX", "stage": args.stage, "task": task_name, "seed": args.seed, "num_envs": args.num_envs, "env_steps": int(env.common_step_counter), "elapsed_s": elapsed, "checkpoint_sha256": digest(args.checkpoint)})
        fields = sorted({key for row in rows for key in row})
        with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "protocol_version": f"g04_g07-{args.stage}-ground-v1",
            "task": task_name,
            "suite": args.suite,
            "backend": "Isaac Lab / PhysX",
            "seed": args.seed,
            "episodes": args.episodes,
            "num_envs": args.num_envs,
            "reset_mode": args.reset_mode,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "config_sha256": digest(args.config),
            "metrics_source": "episodes.csv",
            "frozen": True,
        }
        (args.output / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print("QUADSMASH_STAGE_EVAL_RESULT=" + json.dumps(metrics, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
