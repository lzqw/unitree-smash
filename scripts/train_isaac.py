#!/usr/bin/env python3
"""Budget-gated native Isaac Lab PPO launcher for G04--G07."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


STAGE_GOAL = {
    "smoke": "G03",
    "jump_recover": "G04",
    "virtual_hit": "G05",
    "real_contact": "G06",
    "jump_smash": "G07",
}
STAGE_NATIVE = {
    "smoke": "jump",
    "jump_recover": "jump",
    "virtual_hit": "virtual_hit",
    "real_contact": "real_contact",
    "jump_smash": "jump_smash",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGE_GOAL), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-env-steps", type=int, required=True)
    parser.add_argument("--max-gpu-hours", type=float, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.max_env_steps <= 0 or args.max_gpu_hours <= 0 or args.num_envs <= 0:
        parser.error("max-env-steps, max-gpu-hours and num-envs must be positive")
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error("use either --init-checkpoint or --resume-checkpoint")

    root = Path(__file__).resolve().parents[1]
    budget = json.loads((root / "configs/budget.json").read_text(encoding="utf-8"))
    goal = STAGE_GOAL[args.stage]
    if goal not in budget.get("authorized_goals", []):
        print(json.dumps({"status": "BLOCKED_PERMISSION", "goal": goal, "message": "Update configs/budget.json only after explicit user GPU-hour authorization."}, ensure_ascii=False))
        return 3
    status = json.loads((root / "status/STATUS.json").read_text())
    dependencies = [f"G{i:02d}" for i in range(int(goal[1:]))]
    failed_dependencies = [g for g in dependencies if status["goals"][g]["state"] != "PASSED"]
    if failed_dependencies:
        print(json.dumps({"status": "BLOCKED_DEPENDENCY", "goal": goal, "dependencies": failed_dependencies}))
        return 5
    allowed_gpu_ids = {str(value) for value in budget.get("allowed_gpu_ids", [])}
    requested_device = str(args.device)
    requested_gpu_id = requested_device.split(":", 1)[1] if requested_device.startswith("cuda:") else None
    if requested_gpu_id is None or requested_gpu_id not in allowed_gpu_ids:
        print(json.dumps({
            "status": "BLOCKED_PERMISSION",
            "goal": goal,
            "message": "The requested GPU must be explicitly listed in configs/budget.json allowed_gpu_ids; an empty list is not all GPUs.",
            "requested_device": requested_device,
            "allowed_gpu_ids": sorted(allowed_gpu_ids),
        }, ensure_ascii=False))
        return 3
    suggested = .5 if goal=="G03" else float(budget.get("suggested_incremental_budgets_gpu_hours", {}).get(goal, 0.0))
    if goal=="G03" and args.max_env_steps>2_000_000:
        parser.error("G03 smoke may not exceed 2000000 interactions")
    if args.max_gpu_hours > suggested:
        parser.error(f"requested budget {args.max_gpu_hours} exceeds configured {goal} cap {suggested}")
    if int(budget.get("max_concurrent_gpu_processes", 1)) == 1:
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        if query.returncode != 0:
            print(json.dumps({
                "status": "BLOCKED_RESOURCE",
                "goal": goal,
                "message": "Unable to verify exclusive GPU ownership; refusing to launch training.",
                "stderr": query.stderr.strip(),
            }, ensure_ascii=False))
            return 4
        active_apps = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        # These are infrastructure processes, not competing training jobs:
        # the MPS daemon brokers CUDA contexts and the desktop agent uses a
        # small render context.  Other compute applications remain blocking.
        infrastructure_markers = ("nvidia-cuda-mps-server", "/usr/local/awesun/bin/awesun_desktop")
        competing_apps = [
            line for line in active_apps
            if not any(marker in line for marker in infrastructure_markers)
        ]
        if competing_apps:
            print(json.dumps({
                "status": "BLOCKED_RESOURCE",
                "goal": goal,
                "message": "GPU has active compute applications and the budget allows only one concurrent process.",
                "requested_device": requested_device,
                "active_compute_apps": competing_apps,
            }, ensure_ascii=False))
            return 4

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run-dir must be new; use a new directory with --resume-checkpoint to preserve prior evidence")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    sys.path.insert(0,str(root))
    from quadsmash.process_identity import write_identity
    write_identity(args.run_dir,os.getpid())
    config = args.config or (root / ("configs/coupled_ppo.json" if args.stage=="smoke" else "configs/jump_ppo_v2c.json" if args.stage == "jump_recover" else "configs/event_ppo.json"))
    config_data = json.loads(config.read_text(encoding="utf-8"))
    rollout = int(config_data["runner"]["num_steps_per_env"])
    iterations = (args.max_env_steps - args.num_envs * 128) // (args.num_envs * rollout)
    if iterations < 1:
        parser.error("max-env-steps is smaller than one complete PPO rollout")
    output = args.run_dir / "native"
    isaac_python = os.environ.get("QUADSMASH_ISAAC_PYTHON", "/home/autodrive/Isaac/env_isaaclab/bin/python")
    if not Path(isaac_python).is_file():
        isaac_python = sys.executable
    cmd = [
        isaac_python,
        str(root / "scripts/isaac_coupled_runner.py"),
        "train",
        "--stage", STAGE_NATIVE[args.stage],
        "--config", str(config),
        "--output", str(output),
        "--num-envs", str(args.num_envs),
        "--iterations", str(iterations),
        "--seed", str(args.seed),
        "--max-gpu-hours", str(args.max_gpu_hours),
        "--device", args.device,
    ]
    cmd.append("--headless")
    if args.init_checkpoint:
        cmd.extend(["--init-checkpoint", str(args.init_checkpoint.resolve())])
    if args.resume_checkpoint:
        cmd.extend(["--resume-checkpoint", str(args.resume_checkpoint.resolve())])
    metadata = {
        "goal": goal,
        "stage": args.stage,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_env_steps": args.max_env_steps,
        "max_gpu_hours": args.max_gpu_hours,
        "iterations": iterations,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        guarded_cmd = [sys.executable, str(root / "scripts/run_gpu_job.py"),
                       "--goal", "G04" if goal=="G03" else goal, "--max-seconds", str(args.max_gpu_hours * 3600),
                       "--env-steps-cap", str(iterations * args.num_envs * rollout + args.num_envs * 128),
                       "--output", str(output), "--require-artifact", "smoke_metrics.json", "--", *cmd]
        child = subprocess.Popen(guarded_cmd, cwd=root)
        def forward_stop(_sig, _frame):
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, forward_stop)
        signal.signal(signal.SIGINT, forward_stop)
        metadata["exit_code"] = child.wait()
        # Kit shutdown can conceal a Python exception behind exit code zero.
        # Require the freshly written completion receipt before publishing.
        receipt = output / "smoke_metrics.json"
        if metadata["exit_code"] == 0 and ((output / "error.txt").exists() or not receipt.is_file()):
            metadata["exit_code"] = 1
            metadata["failure"] = "Native runner did not produce a successful completion receipt"
        native_checkpoint = output / "checkpoint.pt"
        if metadata["exit_code"] == 0:
            try:
                report = json.loads(receipt.read_text())
                valid = (report["mode"] == "train" and report["completed_iterations"] > 0
                         and report["numerical_failures"] == 0 and report["env_contract_ok"]
                         and report["checkpoint_sha256"] == hashlib.sha256(native_checkpoint.read_bytes()).hexdigest())
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                metadata["exit_code"] = 1
                metadata["failure"] = "Native training receipt is incomplete, nonfinite, or checkpoint hash does not match"
        if metadata["exit_code"] == 0 and native_checkpoint.is_file() and native_checkpoint.stat().st_size > 0:
            # Keep the checkpoint referenced by independent evaluation intact.
            # The stage finalizer publishes policy.pt after measuring it.
            artifact = root / "artifacts" / goal / ("policy.pt" if goal=="G03" else "training_latest.pt")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native_checkpoint, artifact)
            metadata["artifact_checkpoint"] = str(artifact)
            metadata["artifact_checkpoint_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    finally:
        metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (args.run_dir / "run.pid").unlink(missing_ok=True)
    return int(metadata["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
