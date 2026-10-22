#!/usr/bin/env python3
"""Independent ground-initialized evaluation for the native jump policy."""
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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--config", type=Path, default=Path("configs/jump_ppo.json"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=7)
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

from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = JumpEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    cfg.episode_length_s = config.get("episode_seconds", 4.0)
    env = JumpEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, config["runner"], log_dir=str(args.output / "ppo_eval"), device=env.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    try:
        obs, _ = wrapped.reset()
        initial_height = env.robot.data.root_pos_w[:, 2].detach().clone()
        ground_init = bool(torch.isfinite(initial_height).all() and ((initial_height > 0.35) & (initial_height < 0.50)).all())
        numerical_failures = int(not bool(torch.isfinite(obs["policy"]).all().item()))
        start = time.perf_counter()
        with torch.inference_mode():
            while len(env.episode_records) < args.episodes:
                action = policy(obs)
                if not torch.isfinite(action).all():
                    numerical_failures += 1
                obs, reward, _, _ = wrapped.step(action)
                numerical_failures += int(not bool(torch.isfinite(obs["policy"]).all().item()))
                numerical_failures += int(not bool(torch.isfinite(reward).all().item()))
        elapsed = time.perf_counter() - start
        records = env.episode_records[: args.episodes]
        for row in records:
            row["ground_init"] = ground_init
            row["numerical_failure"] = False
        successes = sum(bool(row["success"]) for row in records)
        metrics = {
            "sample_count": len(records),
            "ground_init_fraction": sum(bool(row["ground_init"]) for row in records) / len(records),
            "jump_recovery_success_rate": successes / len(records),
            "numerical_failures": numerical_failures,
            "backend": "Isaac Lab / PhysX",
            "reset_mode": "ground",
            "num_envs": args.num_envs,
            "seed": args.seed,
            "elapsed_s": elapsed,
            "env_steps": int(env.common_step_counter),
            "checkpoint_sha256": sha256(args.checkpoint),
        }
        with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode", "env_id", "ground_init", "max_height_m", "airborne_s", "recovery_s", "success", "numerical_failure"])
            writer.writeheader()
            for index, row in enumerate(records):
                writer.writerow({"episode": index, **row})
        manifest = {
            "protocol_version": "g04-jump-ground-v1",
            "task": "jump_recover_ground_v1",
            "backend": "Isaac Lab / PhysX",
            "seed": args.seed,
            "episodes": args.episodes,
            "num_envs": args.num_envs,
            "reset_mode": "ground",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "config_sha256": sha256(args.config),
            "metrics_source": "reports/G04/episodes.csv",
            "frozen": True,
        }
        (args.output / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print("QUADSMASH_JUMP_EVAL_RESULT=" + json.dumps(metrics, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
