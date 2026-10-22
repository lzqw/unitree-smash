#!/usr/bin/env python3
"""Frozen, ground-initialized evaluation protocol for staged checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main_numpy() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--reset-mode", choices=("ground",), default="ground")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        parser.error("episodes and num-envs must be positive")
    import numpy as np

    from quadsmash.rl import WholeBodyPPO
    from train import resolve_stage

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    stage = resolve_stage(args.stage)
    agent = WholeBodyPPO.load(checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": 1, "suite": args.suite, "stage": stage.value, "episodes": args.episodes, "reset_mode": args.reset_mode, "seed": 1907, "ground_initialization_only": True, "checkpoint_sha256": digest(checkpoint), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "eval_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    remaining = args.episodes
    episode_id = 0
    while remaining:
        count = min(args.num_envs, remaining)
        env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=count, seed=1907 + episode_id, curriculum_stage=stage))
        obs, _ = env.reset(seed=1907 + episode_id)
        for _ in range(env.cfg.max_steps + 1):
            obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            if np.all(terminated | truncated):
                break
        for local_id, metrics in enumerate(env.log_episode_metrics()):
            events = metrics.pop("events")
            row = {"episode": episode_id + local_id, "stage": stage.value, **metrics, **{f"event_{key}": value for key, value in events.items()}}
            rows.append(row)
        episode_id += count
        remaining -= count
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    contact = np.array([bool(row.get("event_contact")) for row in rows])
    airborne = np.array([bool(row.get("event_airborne_contact")) for row in rows])
    valid = np.array([bool(row.get("event_valid_return")) for row in rows])
    recovery = np.array([bool(row.get("event_stable_recovery")) for row in rows])
    success = np.array([bool(row.get("event_success")) for row in rows])
    metrics = {"episodes": len(rows), "contact_rate": float(contact.mean()), "airborne_contact_rate": float(airborne.mean()), "valid_return_rate": float(valid.mean()), "stable_recovery_rate": float(recovery.mean()), "success_rate": float(success.mean()), "conditioned_recovery_given_airborne_contact": float(recovery[airborne].mean()) if np.any(airborne) else 0.0, "numerical_failures": 0}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": protocol, "metrics": metrics, "output": str(output)}, indent=2, ensure_ascii=False))


def main_native() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("jump_recover", "virtual_hit", "real_contact", "jump_smash"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--reset-mode", choices=("ground",), default="ground")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=600.)
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0:
        parser.error("episodes and num-envs must be positive")
    root = Path(__file__).resolve().parents[1]
    budget=json.loads((root/'configs/budget.json').read_text())
    goal={'jump_recover':'G04','virtual_hit':'G05','real_contact':'G06','jump_smash':'G07'}[args.stage]
    if goal not in budget['authorized_goals'] or args.device!='cuda:0' or 0 not in budget['allowed_gpu_ids']:
        parser.error('stage or GPU is outside the existing authorization')
    status=json.loads((root/'status/STATUS.json').read_text())
    if any(status['goals'][f'G{i:02d}']['state']!='PASSED' for i in range(int(goal[1:]))):
        parser.error('stage dependencies have not passed')
    if args.stage=='jump_recover' and args.episodes!=400:
        parser.error('G04 requires both frozen panels: --episodes 400')
    isaac_python = os.environ.get("QUADSMASH_ISAAC_PYTHON", "/home/autodrive/Isaac/env_isaaclab/bin/python")
    if not Path(isaac_python).is_file():
        isaac_python = sys.executable
    stage = {"jump_recover": "jump", "virtual_hit": "virtual_hit", "real_contact": "real_contact", "jump_smash": "jump_smash"}[args.stage]
    config = args.config or (root / ("configs/jump_ppo_v2c.json" if args.stage == "jump_recover" else "configs/event_ppo.json"))
    cmd = [
        isaac_python,
        str(root / ('scripts/isaac_jump_evaluate_v2.py' if args.stage=='jump_recover' else 'scripts/isaac_stage_evaluate.py')),
        "--stage", stage,
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--config", str(Path(config).resolve()),
        "--output", str(Path(args.output).resolve()),
        "--episodes", str(args.episodes),
        "--num-envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--suite", args.suite,
        "--reset-mode", args.reset_mode,
        "--device", args.device,
        "--headless",
    ]
    if args.stage!='jump_recover':
        parser.error('G05+ native evaluators require their stage-specific protocol audit before execution')
    import math
    seconds=json.loads(Path(config).read_text())['episode_seconds']
    cap=2*math.ceil(200/args.num_envs)*args.num_envs*math.ceil(seconds/.01)
    guarded=[sys.executable,str(root/'scripts/run_gpu_job.py'),'--goal',goal,
        '--max-seconds',str(args.max_seconds),'--env-steps-cap',str(cap),
        '--output',str(Path(args.output).resolve()),'--require-artifact','metrics.json','--',*cmd]
    return subprocess.run(guarded, cwd=root, check=False).returncode


if __name__ == "__main__":
    argv = sys.argv[1:]
    backend = "isaac"
    if "--backend" in argv:
        index = argv.index("--backend")
        if index + 1 >= len(argv):
            raise SystemExit("--backend requires isaac or numpy")
        backend = argv[index + 1]
        del argv[index : index + 2]
        sys.argv = [sys.argv[0], *argv]
    if backend == "isaac":
        raise SystemExit(main_native())
    if backend == "numpy":
        main_numpy()
    else:
        raise SystemExit(f"unsupported backend: {backend}")
