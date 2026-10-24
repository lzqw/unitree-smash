#!/usr/bin/env python3
"""Run one project GPU job with a cumulative ledger, deadline and receipts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
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


def authorization_family(windows, name):
    """Continuations inherit the original clock and cumulative resource cap."""
    seen = set()
    while windows[name].get("continuation_of"):
        if name in seen:
            raise ValueError("cyclic authorization continuation")
        seen.add(name)
        name = windows[name]["continuation_of"]
        if name not in windows:
            raise ValueError("unknown original authorization")
    return name


def check_window_budget(budget, ledger, name, seconds, step_cap, now):
    windows = budget["authorization_windows"]
    original = authorization_family(windows, name)
    family = {key for key in windows if authorization_family(windows, key) == original}
    all_records = [r for r in ledger["runs"] if r.get("budget_window") in family]
    # Both the local continuation allocation and the original family cap apply.
    for key in dict.fromkeys((name, original)):
        records = all_records if key == original else [
            r for r in all_records if r.get("budget_window") == key]
        config = windows[key]
        hours = sum(r["charged_gpu_hours"] for r in records)
        if config["max_gpu_hours"] is not None and hours + seconds / 3600 > config["max_gpu_hours"]:
            raise ValueError(f"cumulative GPU budget exceeds {config['max_gpu_hours']} h in {key}; charged {hours:.4f} h")
        if config.get("max_env_steps") is not None:
            steps = sum(r["charged_env_steps"] for r in records)
            if steps + step_cap > config["max_env_steps"]:
                raise ValueError("cumulative environment step budget exceeds original authorization")
    starts = [datetime.fromisoformat(r["started_at"]) for r in all_records if r.get("started_at")]
    state = ledger.get("budget_windows", {}).get(original)
    if state:
        starts.append(datetime.fromisoformat(state["started_at"]))
    started = min(starts) if starts else now
    wallclock_hours = windows[original]["max_wallclock_hours"]
    if wallclock_hours is None:
        return original, started, None
    deadline = started.timestamp() + wallclock_hours * 3600
    if now.timestamp() + seconds > deadline:
        raise ValueError(f"original authorization {original} wall-clock deadline is {datetime.fromtimestamp(deadline, timezone.utc).isoformat()}; requested job would exceed it")
    return original, started, datetime.fromtimestamp(deadline, timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", default="G04")
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--env-steps-cap", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-artifact", required=True)
    parser.add_argument("--budget-window", default="legacy-G04")
    parser.add_argument("--paused-external-pid", type=int, action="append", default=[],
                        help="Explicitly authorized temporarily suspended jobs; must be owned by this UID and in stopped state")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.max_seconds) or args.max_seconds < 35 or args.env_steps_cap < 0:
        parser.error("command, max-seconds >= 35 and nonnegative env-steps-cap are required")
    budget = json.loads((ROOT / "configs/budget.json").read_text())
    windows = budget.get("authorization_windows", {})
    if args.budget_window != "legacy-G04" and args.budget_window not in windows:
        parser.error("unknown authorization window")
    if args.budget_window != "legacy-G04" and args.goal not in windows[args.budget_window]["goals"]:
        parser.error("goal is not authorized by this budget window")
    if 0 not in budget["allowed_gpu_ids"]:
        parser.error("GPU0 is not authorized")
    if args.budget_window == "legacy-G04" and args.goal not in budget["authorized_goals"]:
        parser.error("goal or GPU0 is not authorized")
    ledger_path = ROOT / "status/GPU_LEDGER.json"
    with (ROOT / "status/gpu_job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text())
        prior = [r for r in ledger["runs"] if r["goal"] == args.goal]
        if any(r["status"] == "RUNNING" for r in prior):
            parser.error("an unresolved RUNNING ledger entry requires process reconciliation")
        window_records = [r for r in ledger["runs"] if r.get("budget_window", "legacy-G04") == args.budget_window]
        hours = sum(r["charged_gpu_hours"] for r in window_records)
        steps = sum(r["charged_env_steps"] for r in window_records)
        window = windows.get(args.budget_window)
        cap = (window["max_gpu_hours"] if window else budget["suggested_incremental_budgets_gpu_hours"][args.goal])
        if cap is not None and hours + args.max_seconds / 3600 > cap:
            parser.error(f"cumulative GPU budget would exceed {cap} h in {args.budget_window}; already charged {hours:.4f} h")
        if window and window.get("max_env_steps") is not None and steps + args.env_steps_cap > window["max_env_steps"]:
            parser.error("cumulative environment step budget would exceed the authorization window")
        if args.budget_window == "legacy-G04" and args.goal == "G04" and steps + args.env_steps_cap > 200_000_000:
            parser.error("cumulative G04 environment step budget would exceed 200000000")
        if window:
            try:
                original_window, original_start, original_deadline = check_window_budget(
                    budget, ledger, args.budget_window, args.max_seconds,
                    args.env_steps_cap, datetime.now(timezone.utc))
            except ValueError as exc:
                parser.error(str(exc))
        apps = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
        paused=[]
        for pid in args.paused_external_pid:
            proc_path=Path(f'/proc/{pid}')
            if proc_path.stat().st_uid!=os.getuid():
                parser.error('paused job must belong to the current user')
            state=(proc_path/'stat').read_text().split(') ',1)[1].split()[0]
            if state not in ('T','t'):
                parser.error(f'explicitly paused PID {pid} is not stopped')
            paused.append(dict(pid=pid,state=state,command=(proc_path/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
        competing = [line for line in apps.splitlines() if not any(s in line for s in ("nvidia-cuda-mps-server", "/usr/local/awesun/bin/awesun_desktop"))
                     and int(line.split(',',1)[0]) not in args.paused_external_pid]
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
        if window:
            windows_state = ledger.setdefault("budget_windows", {})
            windows_state.setdefault(args.budget_window, {"started_at": datetime.now(timezone.utc).isoformat(), "goals": window["goals"]})
        record = {"goal": args.goal, "budget_window": args.budget_window, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "command": command, "output": str(output), "gpu": 0,
                  "status": "RUNNING", "charged_gpu_hours": args.max_seconds / 3600,
                  "charged_env_steps": args.env_steps_cap,
                  "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
        if window:
            record.update(original_authorization=original_window,
                          authorization_started_at=original_start.isoformat(),
                          authorization_deadline=original_deadline.isoformat() if original_deadline else None)
        if paused:record['explicitly_paused_external_jobs']=paused
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
