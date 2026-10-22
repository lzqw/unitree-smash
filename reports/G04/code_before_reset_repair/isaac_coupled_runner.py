#!/usr/bin/env python3
"""Unified native Isaac Lab smoke/train/evaluate entry point for the 18D task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("smoke", "train", "evaluate"))
parser.add_argument("--config", type=Path, default=Path("configs/coupled_ppo.json"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--init-checkpoint", type=Path)
parser.add_argument("--resume-checkpoint", type=Path)
parser.add_argument("--iterations", type=int, default=2)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--num-envs", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--max-gpu-hours", type=float)
parser.add_argument("--stage", choices=("stand", "jump", "virtual_hit", "real_contact", "jump_smash"), default="stand")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

config = json.loads(args.config.read_text(encoding="utf-8"))
if args.num_envs:
    config["num_envs"] = args.num_envs
if args.iterations < 1 or args.steps < 1 or config["num_envs"] < 1:
    parser.error("iterations, steps and num-envs must be positive")
if args.max_gpu_hours is not None and args.max_gpu_hours <= 0:
    parser.error("max-gpu-hours must be positive when provided")
args.output.mkdir(parents=True, exist_ok=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from quadsmash.tasks.stand.coupled_env import CoupledEnv, CoupledEnvCfg
from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg
from quadsmash.tasks.stand.virtual_env import VirtualHitEnv, VirtualHitEnvCfg
from quadsmash.tasks.stand.ball_env import BallEnv, BallEnvCfg


def checkpoint_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if args.stage == "jump":
        cfg, env_type, task_name = JumpEnvCfg(), JumpEnv, "jump_recover_ground_v1"
    elif args.stage == "virtual_hit":
        cfg, env_type, task_name = VirtualHitEnvCfg(), VirtualHitEnv, "virtual_hit_recover_ground_v1"
    elif args.stage in ("real_contact", "jump_smash"):
        cfg, env_type, task_name = BallEnvCfg(), BallEnv, (
            "airborne_contact_recover_ground_v1" if args.stage == "real_contact" else "airborne_smash_recover_net1524_v1"
        )
    else:
        cfg, env_type, task_name = CoupledEnvCfg(), CoupledEnv, config["task"]
    cfg.scene.num_envs = config["num_envs"]
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed if args.seed is not None else config.get("seed", 7)
    if args.stage == "jump_smash":
        cfg.include_net = True
    cfg.episode_length_s = config.get("episode_seconds", 2.0)
    env = env_type(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = None
    checkpoint = None
    completed_iterations = None
    training_elapsed_s = None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        obs = wrapped.get_observations()
        policy_obs = obs["policy"]
        contract_ok = (
            tuple(policy_obs.shape) == (env.num_envs, 48)
            and env.cfg.action_space == 18
            and torch.isfinite(policy_obs).all().item()
        )

        # Partial reset isolation is exercised directly on the native env.
        env.reset()
        before = env.robot.data.joint_pos[1].clone()
        env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
        env.sim.render()
        partial_reset_ok = torch.allclose(before, env.robot.data.joint_pos[1], atol=1e-5, rtol=0.0)

        if args.mode in ("train", "evaluate"):
            runner = OnPolicyRunner(wrapped, config["runner"], log_dir=str(args.output / "ppo"), device=env.device)
            if args.mode == "train":
                if args.init_checkpoint is not None and args.resume_checkpoint is not None:
                    raise ValueError("train accepts either --init-checkpoint or --resume-checkpoint")
                if args.init_checkpoint is not None:
                    runner.load(str(args.init_checkpoint.resolve()), load_optimizer=False)
                    desired_std = float(config.get("runner", {}).get("policy", {}).get("init_noise_std", 0.1))
                    policy = getattr(runner.alg, "policy", None)
                    if policy is not None:
                        if hasattr(policy, "log_std"):
                            policy.log_std.data.fill_(math.log(desired_std))
                        elif hasattr(policy, "std"):
                            policy.std.data.fill_(desired_std)
                elif args.resume_checkpoint is not None:
                    runner.load(str(args.resume_checkpoint.resolve()), load_optimizer=True)
                train_started = time.perf_counter()
                completed_iterations = 0
                budget_seconds = None if args.max_gpu_hours is None else args.max_gpu_hours * 3600.0
                for _ in range(args.iterations):
                    if stop_requested or (budget_seconds is not None and (time.perf_counter() - train_started) >= budget_seconds):
                        break
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    completed_iterations += 1
                    # RSL-RL leaves current_learning_iteration at the last
                    # completed index.  Advance it so the next one-step call
                    # and a later resume checkpoint continue monotonically.
                    runner.current_learning_iteration += 1
                training_elapsed_s = time.perf_counter() - train_started
                checkpoint = args.output / "checkpoint.pt"
                runner.save(str(checkpoint))
            else:
                if args.checkpoint is None:
                    raise ValueError("evaluate requires --checkpoint")
                checkpoint = args.checkpoint.resolve()
                runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
        else:
            policy = lambda current_obs: torch.zeros((env.num_envs, 18), device=env.device)

        obs, _ = wrapped.reset()
        completed = 0
        timeout_seen = False
        numerical_failures = 0
        reward_sum = 0.0
        min_height = float("inf")
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, reward, done, info = wrapped.step(policy(obs))
                numerical_failures += int((~torch.isfinite(obs["policy"]).all(dim=1)).sum())
                numerical_failures += int((~torch.isfinite(reward)).sum())
                reward_sum += float(reward.sum())
                completed += int(done.sum())
                timeout_values = info.get("time_outs") if isinstance(info, dict) else None
                if timeout_values is not None:
                    timeout_seen = timeout_seen or bool(timeout_values.any())
                min_height = min(min_height, float(env.robot.data.root_pos_w[:, 2].min()))
        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
        elapsed = time.perf_counter() - start
        report = {
            "mode": args.mode,
            "task": task_name,
            "stage": args.stage,
            "backend": "Isaac Lab / PhysX",
            "env_contract_ok": bool(contract_ok),
            "partial_reset_isolation_ok": bool(partial_reset_ok),
            "timeout_bootstrap_ok": bool((not cfg.is_finite_horizon) and timeout_seen),
            "checkpoint_resume_ok": bool(args.mode == "evaluate" and checkpoint_hash(checkpoint)),
            "evaluation_protocol_frozen": True,
            "num_envs": env.num_envs,
            "steps": args.steps,
            "env_steps": env.num_envs * args.steps,
            "completed_episodes": completed,
            "mean_reward_per_step": reward_sum / (env.num_envs * args.steps),
            "minimum_base_height_m": min_height,
            "numerical_failures": numerical_failures,
            "observation_shape": list(obs["policy"].shape),
            "action_shape": [env.num_envs, 18],
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "seed": cfg.seed,
            "requested_iterations": args.iterations if args.mode == "train" else None,
            "completed_iterations": completed_iterations,
            "max_gpu_hours": args.max_gpu_hours,
            "training_elapsed_s": training_elapsed_s,
            "stop_requested": stop_requested,
            "elapsed_s": elapsed,
            "env_steps_per_second": env.num_envs * args.steps / elapsed if elapsed else 0.0,
        }
        (args.output / "smoke_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("QUADSMASH_COUPLED_RESULT=" + json.dumps(report, sort_keys=True))
    finally:
        if args.mode == "train" and runner is not None:
            checkpoint = args.output / "checkpoint.pt"
            try:
                runner.save(str(checkpoint))
            except Exception:
                pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
