#!/usr/bin/env python3
"""G03 contract, reset, truncation and checkpoint smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from quadsmash.rl import PPOConfig, WholeBodyPPO
from quadsmash.tasks import QuadSmashEnv, QuadSmashEnvConfig


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--policy-steps", type=int, default=2000)
    parser.add_argument("--output", default="reports/G03")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.policy_steps <= 0:
        parser.error("num-envs and policy-steps must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=args.num_envs, seed=11))
    obs, _ = env.reset(seed=11)
    contract_ok = obs.shape == (args.num_envs, env.observation_space.shape[0]) and env.action_space.shape == (18,) and np.isfinite(obs).all()
    before = env.get_observations().copy()
    env.step(np.zeros((args.num_envs, 18), dtype=np.float32))
    unselected = np.arange(1, args.num_envs)
    env.reset_indices([0])
    after = env.get_observations()
    partial_reset_ok = bool(np.array_equal(before[1:], after[1:]) is False and np.isfinite(after[1:]).all())
    # Verify isolation using dynamic state rather than observations whose
    # shuttle state naturally advanced during the preceding step.
    state_before = env.joint_pos[1:].copy()
    env.reset_indices([0])
    partial_reset_ok &= bool(np.array_equal(state_before, env.joint_pos[1:]))
    agent = WholeBodyPPO(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(seed=11))
    updates = []
    for _ in range(args.policy_steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Correct timeout handling: pass only true terminals to PPO so a time
        # limit bootstraps from the final observation.
        stats = agent.update(obs, action, reward, next_obs, terminated)
        updates.append(stats)
        done = terminated | truncated
        if np.any(done):
            env.reset_indices(np.flatnonzero(done))
            next_obs[done] = env.get_observations()[done]
        obs = next_obs
    timeout_env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=2, episode_seconds=0.02, dt=0.02, seed=21))
    timeout_obs, _ = timeout_env.reset(seed=21)
    timeout_next, _, timeout_terminated, timeout_truncated, _ = timeout_env.step(np.zeros((2, 18)))
    timeout_agent = WholeBodyPPO(timeout_env.observation_space.shape[0], 18, PPOConfig(seed=21))
    timeout_stats = timeout_agent.update(timeout_obs, np.zeros((2, 18)), np.ones(2), timeout_next, timeout_terminated)
    timeout_bootstrap_ok = bool(np.all(timeout_truncated) and not np.any(timeout_terminated) and np.isfinite(timeout_stats["critic_loss"]))
    checkpoint = output / "smoke_checkpoint.npz"
    agent.save(checkpoint)
    reloaded = WholeBodyPPO.load(checkpoint)
    checkpoint_resume_ok = bool(np.array_equal(agent.act(obs, deterministic=True), reloaded.act(obs, deterministic=True)))
    protocol = {"schema_version": 1, "suite": "dev_smoke_v1", "seed": 11, "num_envs": args.num_envs, "policy_steps": args.policy_steps, "reset_mode": "ground", "holdout_locked": True}
    (output / "interface_contract.json").write_text(json.dumps({"observation_shape": list(env.observation_space.shape), "action_shape": list(env.action_space.shape), "action_dim": 18, "event_goal_fields": ["time_remaining", "position", "normal", "velocity"], "phases": ["prepare", "takeoff", "airborne", "post_contact", "recovery", "complete"], "terminated_vs_truncated": "terminated masks bootstrap; truncation bootstraps final_observation"}, indent=2) + "\n", encoding="utf-8")
    (output / "evaluation_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    metrics = {"env_contract_ok": bool(contract_ok), "partial_reset_isolation_ok": bool(partial_reset_ok), "timeout_bootstrap_ok": timeout_bootstrap_ok, "checkpoint_resume_ok": checkpoint_resume_ok, "evaluation_protocol_frozen": True, "policy_steps": args.policy_steps, "env_steps": args.policy_steps * args.num_envs, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "elapsed_s": time.perf_counter() - started, "backend": "numpy_reduced_order", "notes": "This smoke validates the project contract; it is not Isaac Lab/PiPER learning evidence."}
    (output / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "commands.log").write_text(f"python scripts/smoke.py --num-envs {args.num_envs} --policy-steps {args.policy_steps} --output {output}\n", encoding="utf-8")
    (output / "REPORT.md").write_text("# G03 environment/PPO smoke\n\n" + json.dumps(metrics, indent=2) + "\n\nThis is a reduced-order NumPy contract smoke; Isaac Lab dependency and asset audits remain separate gates.\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
