#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadsmash.rl import LinearPPO, PPOConfig, WholeBodyPPO
from quadsmash.tasks import QuadSmashEnv, QuadSmashEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint", default="checkpoints/ppo.npz")
    parser.add_argument("--metrics", default="reports/train.json")
    parser.add_argument("--algorithm", choices=("whole_body", "linear"), default="whole_body")
    parser.add_argument("--curriculum-stage", default="real_contact")
    args = parser.parse_args()
    env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=args.num_envs, curriculum_stage=args.curriculum_stage))
    config = PPOConfig(seed=7)
    agent = WholeBodyPPO(env.observation_space.shape[0], env.action_space.shape[0], config) if args.algorithm == "whole_body" else LinearPPO(env.observation_space.shape[0], env.action_space.shape[0], config)
    obs, _ = env.reset()
    logs = []
    for step in range(args.steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        logs.append({"step": step, **agent.update(obs, action, reward, next_obs, terminated | truncated)})
        obs = next_obs
        if np.all(terminated | truncated):
            obs, _ = env.reset()
    agent.save(args.checkpoint)
    metrics = {"algorithm": "whole_body_ppo" if args.algorithm == "whole_body" else "linear_ppo", "num_envs": args.num_envs, "steps": args.steps, "checkpoint": str(args.checkpoint), "curriculum_stage": args.curriculum_stage, "logs": logs}
    output = Path(args.metrics); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

