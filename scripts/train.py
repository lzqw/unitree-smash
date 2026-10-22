#!/usr/bin/env python3
"""Budgeted event-conditioned PPO training entry point.

Staged G04--G07 commands dispatch to the native Isaac Lab runner.  The
dependency-light NumPy backend remains available explicitly for CI/API work
and is not a source of Isaac Lab training evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

STAGES = {
    "jump_recover": "jump_recovery",
    "jump_recovery": "jump_recovery",
    "virtual_hit": "virtual_event",
    "virtual_event": "virtual_event",
    "real_contact": "real_contact",
    "jump_smash": "smash_and_recover",
    "smash_and_recover": "smash_and_recover",
    "domain_randomized": "domain_randomized",
    "smoke": "real_contact",
}

NATIVE_STAGES = {"smoke", "jump_recover", "jump_recovery", "virtual_hit", "virtual_event", "real_contact", "jump_smash", "smash_and_recover"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_stage(value: str):
    from quadsmash.tasks import CurriculumStage

    try:
        return CurriculumStage(STAGES[value])
    except KeyError as exc:
        raise ValueError(f"unsupported stage {value!r}; choose one of {sorted(STAGES)}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-env-steps", type=int, required=True)
    parser.add_argument("--max-gpu-hours", type=float, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=128)
    args = parser.parse_args()

    # Keep the public/native CLI usable from the system interpreter.  The
    # reduced NumPy backend is imported only after argparse has selected it;
    # native Isaac Lab stages are dispatched before this function is called.
    import numpy as np
    from quadsmash.rl import PPOConfig, WholeBodyPPO
    from quadsmash.tasks import QuadSmashEnv, QuadSmashEnvConfig

    if args.max_env_steps <= 0 or args.max_gpu_hours < 0 or args.num_envs <= 0:
        parser.error("max-env-steps and num-envs must be positive; max-gpu-hours cannot be negative")
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error("use either --init-checkpoint or --resume-checkpoint, not both")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    stage = resolve_stage(args.stage)
    env = QuadSmashEnv(QuadSmashEnvConfig(num_envs=args.num_envs, seed=args.seed, curriculum_stage=stage))
    checkpoint = args.resume_checkpoint or args.init_checkpoint
    if args.resume_checkpoint:
        agent = WholeBodyPPO.load(args.resume_checkpoint)
        mode = "resume"
    else:
        agent = WholeBodyPPO(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig(seed=args.seed))
        mode = "init" if args.init_checkpoint else "fresh"
        if args.init_checkpoint:
            source = WholeBodyPPO.load(args.init_checkpoint)
            if source.obs_dim != agent.obs_dim or source.action_dim != agent.action_dim:
                raise ValueError("init checkpoint observation/action contract is incompatible")
            agent.policy.shared = [value.copy() for value in source.policy.shared]
            agent.policy.leg_head = source.policy.leg_head.copy(); agent.policy.arm_head = source.policy.arm_head.copy()
            agent.policy.leg_bias = source.policy.leg_bias.copy(); agent.policy.arm_bias = source.policy.arm_bias.copy(); agent.policy.log_std = source.policy.log_std.copy()
    obs, reset_info = env.reset(seed=args.seed)
    started = time.perf_counter()
    started_at = now_iso()
    stop_requested = {"value": False}
    def request_stop(signum, _frame):
        stop_requested["value"] = True
        (run_dir / "stop.requested").write_text(f"signal={signum}\n", encoding="utf-8")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    logs = []
    total_env_steps = 0
    update = 0
    last_checkpoint = None
    try:
        while total_env_steps < args.max_env_steps and not stop_requested["value"]:
            actions = agent.act(obs)
            next_obs, reward, terminated, truncated, infos = env.step(actions)
            # Truncation bootstraps from final_observation; only true terminal
            # failures are passed as terminal masks to PPO.
            stats = agent.update(obs, actions, reward, next_obs, terminated)
            update += 1
            total_env_steps += env.num_envs
            elapsed = time.perf_counter() - started
            logs.append({"update": update, "env_steps": total_env_steps, "elapsed_s": elapsed, **stats, "terminated": int(terminated.sum()), "truncated": int(truncated.sum())})
            done = terminated | truncated
            if np.any(done):
                env.reset_indices(np.flatnonzero(done))
                next_obs[done] = env.get_observations()[done]
            obs = next_obs
            if update % max(1, args.checkpoint_interval) == 0 or total_env_steps >= args.max_env_steps or stop_requested["value"]:
                last_checkpoint = run_dir / "checkpoint.npz"
                agent.save(last_checkpoint)
                (run_dir / "checkpoint.meta.json").write_text(json.dumps({"stage": stage.value, "seed": args.seed, "env_steps": total_env_steps, "mode": mode, "saved_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
    finally:
        if last_checkpoint is None:
            last_checkpoint = run_dir / "checkpoint.npz"
            agent.save(last_checkpoint)
        (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(item, allow_nan=False) for item in logs) + ("\n" if logs else ""), encoding="utf-8")
        result = {"stage": stage.value, "backend": "numpy_reduced_order", "seed": args.seed, "num_envs": args.num_envs, "env_steps": total_env_steps, "updates": update, "max_env_steps": args.max_env_steps, "max_gpu_hours": args.max_gpu_hours, "elapsed_s": time.perf_counter() - started, "stop_requested": stop_requested["value"], "checkpoint": str(last_checkpoint), "checkpoint_mode": mode, "reset_info": reset_info, "metrics_path": str(run_dir / "metrics.jsonl"), "protocol_note": "This is not an Isaac Lab/PiPER training claim; G00-G03 dependency audit remains authoritative."}
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        try:
            (run_dir / "run.pid").unlink()
        except FileNotFoundError:
            pass
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # The public staged-training CLI is native Isaac Lab by default.  The
    # reduced NumPy backend remains available explicitly for CI/API work with
    # ``--backend numpy`` and is never valid as a G04--G07 evidence source.
    argv = sys.argv[1:]
    stage_value = None
    for index, value in enumerate(argv):
        if value == "--stage" and index + 1 < len(argv):
            stage_value = argv[index + 1]
            break
    backend = "isaac"
    if "--backend" in argv:
        backend_index = argv.index("--backend")
        if backend_index + 1 < len(argv):
            backend = argv[backend_index + 1]
        del argv[backend_index : backend_index + 2]
        sys.argv = [sys.argv[0], *argv]
    if backend == "isaac" and stage_value in NATIVE_STAGES:
        from train_isaac import main as native_main

        raise SystemExit(native_main())
    main()
