#!/usr/bin/env python3
"""Run one project GPU job with a cumulative ledger, deadline and receipts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="G04")
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--env-steps-cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-artifact", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_seconds < 35 or args.env_steps_cap < 0:
        parser.error("command, max-seconds >= 35 and nonnegative env-steps-cap are required")
    budget = json.loads((ROOT / "configs/budget.json").read_text())
    if args.goal not in budget["authorized_goals"] or 0 not in budget["allowed_gpu_ids"]:
        parser.error("goal or GPU0 is not authorized")
    ledger_path = ROOT / "status/GPU_LEDGER.json"
    with (ROOT / "status/gpu_job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text())
        prior = [r for r in ledger["runs"] if r["goal"] == args.goal]
        if any(r["status"] == "RUNNING" for r in prior):
            parser.error("an unresolved RUNNING ledger entry requires process reconciliation")
        hours = sum(r["charged_gpu_hours"] for r in prior)
        steps = sum(r["charged_env_steps"] for r in prior)
        cap = budget["suggested_incremental_budgets_gpu_hours"][args.goal]
        if hours + args.max_seconds / 3600 > cap:
            parser.error(f"cumulative GPU budget would exceed {cap} h; already charged {hours:.4f} h")
        if args.goal == "G04" and steps + args.env_steps_cap > 200_000_000:
            parser.error("cumulative G04 environment step budget would exceed 200000000")
        apps = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
        competing = [line for line in apps.splitlines() if not any(s in line for s in ("nvidia-cuda-mps-server", "/usr/local/awesun/bin/awesun_desktop"))]
        if competing:
            parser.error("GPU0 has competing compute processes: " + repr(competing))
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            parser.error("output must be new to prevent accepting a stale receipt")
        output.mkdir(parents=True, exist_ok=True)
        files = sorted([*ROOT.glob("quadsmash/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("configs/*.json"), *ROOT.glob("assets/*.usd")])
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        atomic_json(output / "source_hashes.json", hashes)
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
            for p in files:
                snapshot.write(p, str(p.relative_to(ROOT)))
        subprocess.run(["git", "diff", "--binary"], cwd=ROOT, stdout=(output / "dirty.patch").open("w"), check=True)
        record = {"goal": args.goal, "command": command, "output": str(output), "gpu": 0,
                  "status": "RUNNING", "charged_gpu_hours": args.max_seconds / 3600,
                  "charged_env_steps": args.env_steps_cap,
                  "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
        ledger["runs"].append(record)
        atomic_json(ledger_path, ledger)
        started = time.monotonic()
        with (output / "stdout.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["child_pid"] = proc.pid
            atomic_json(ledger_path, ledger)
            def stop(_sig, _frame):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            timed_out = False
            try:
                code = proc.wait(timeout=max(1.0, args.max_seconds - 30))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
        receipt = output / args.require_artifact
        error_files = list(output.rglob("error.txt"))
        okay = code == 0 and receipt.is_file() and not error_files and not timed_out
        elapsed = time.monotonic() - started
        record.update(status="SUCCEEDED" if okay else "FAILED", exit_code=code,
                      timed_out=timed_out, receipt_present=receipt.is_file(),
                      elapsed_s=elapsed, charged_gpu_hours=elapsed / 3600,
                      finished_at=datetime.now(timezone.utc).isoformat())
        # Conservatively charge the declared interaction cap, including failed jobs.
        atomic_json(output / "job.json", record)
        atomic_json(ledger_path, ledger)
        with (ROOT / "status/RUNS.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
