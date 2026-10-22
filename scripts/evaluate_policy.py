#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadsmash.rl import LinearPPO, WholeBodyPPO
from quadsmash.tasks import QuadSmashEnv, QuadSmashEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", default="reports/eval.json")
    args = parser.parse_args()
    metadata_path = Path(args.checkpoint).with_suffix(".json")
    algorithm = "linear_ppo"
    if metadata_path.exists():
        algorithm = json.loads(metadata_path.read_text(encoding="utf-8")).get("algorithm", algorithm)
    agent = WholeBodyPPO.load(args.checkpoint) if algorithm == "whole_body_ppo" else LinearPPO.load(args.checkpoint)
    env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=args.episodes, seed=19))
    obs, _ = env.reset(seed=19)
    for _ in range(env.cfg.max_steps):
        obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
        if np.all(terminated | truncated):
            break
    metrics = {"episodes": args.episodes, "checkpoint": args.checkpoint, "algorithm": algorithm, "episode_metrics": env.log_episode_metrics()}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

