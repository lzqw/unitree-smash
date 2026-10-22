"""P0 native simulation, RSL-RL PPO training and separate-process evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["smoke", "train", "evaluate", "stand"])
parser.add_argument("--config", default="configs/p0.json")
parser.add_argument("--output", required=True)
parser.add_argument("--checkpoint")
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num-envs", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
if args.num_envs:
    config["num_envs"] = args.num_envs
if args.steps < 1 or args.iterations < 1 or config["num_envs"] < 1:
    parser.error("steps, iterations and num-envs must be positive")
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.go2_env import Go2DiagnosticEnv, Go2DiagnosticCfg, JOINT_ORDER


def main():
    cfg = Go2DiagnosticCfg()
    cfg.seed = config["seed"]
    cfg.scene.num_envs = config["num_envs"]
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.dt = config["dt"]
    cfg.decimation = config["decimation"]
    cfg.episode_length_s = config["episode_seconds"]
    cfg.action_scale = config["action_scale"]
    cfg.robot = cfg.robot.copy()
    cfg.robot.spawn.usd_path = config["go2_usd"]
    cfg.robot.spawn.articulation_props.enabled_self_collisions = True
    env = Go2DiagnosticEnv(cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapped.get_observations()
        assert obs["policy"].shape == (cfg.scene.num_envs, 45)
        assert torch.isfinite(obs["policy"]).all()
        checkpoint = None
        runner = None
        if args.mode in ("train", "evaluate"):
            runner = OnPolicyRunner(wrapped, config["runner"], log_dir=str(output / "ppo"), device=env.device)
            if args.mode == "train":
                runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
                checkpoint = output / "checkpoint.pt"
                runner.save(str(checkpoint))
            else:
                if not args.checkpoint:
                    raise ValueError("evaluate requires --checkpoint")
                checkpoint = Path(args.checkpoint).resolve()
                runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
        else:
            policy = lambda obs: torch.zeros((env.num_envs, 12), device=env.device)

        obs, _ = wrapped.reset()
        failures = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        episodes = 0
        total_reward = 0.0
        min_height, max_speed, max_effort = float("inf"), 0.0, 0.0
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, reward, done, info = wrapped.step(policy(obs))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert reward.shape == done.shape == (env.num_envs,)
                failures += env.reset_terminated.long()
                episodes += int(done.sum())
                total_reward += float(reward.sum())
                min_height = min(min_height, float(env.robot.data.root_pos_w[:, 2].min()))
                max_speed = max(max_speed, float(env.robot.data.joint_vel.abs().max()))
                max_effort = max(max_effort, float(env.robot.data.applied_torque.abs().max()))
        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
        elapsed = time.perf_counter() - start
        report = {
            "mode": args.mode, "task": config["task"], "backend": "Isaac Lab / PhysX",
            "scope": "12-DOF bare Go2 runtime diagnostic; no loaded jump or smash claim",
            "num_envs": env.num_envs, "steps": args.steps, "elapsed_s": elapsed,
            "env_steps_per_second": env.num_envs * args.steps / elapsed,
            "completed_episodes": episodes, "failure_count": int(failures.sum()),
            "mean_reward_per_step": total_reward / (env.num_envs * args.steps),
            "minimum_base_height_m": min_height, "maximum_joint_speed_rad_s": max_speed,
            "maximum_effort_nm": max_effort, "all_tensors_finite": True,
            "observation_shape": list(obs["policy"].shape), "action_shape": [env.num_envs, 12],
            "joint_order": list(JOINT_ORDER), "body_names": env.robot.body_names,
            "mass_per_body_kg": env.robot.root_physx_view.get_masses()[0].tolist(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
            "config": config,
        }
        if runner:
            assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
            report["policy_parameters_finite"] = True
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("QUADSMASH_RESULT=" + json.dumps(report))
    finally:
        env.close()


try:
    main()
finally:
    app.close()
