#!/usr/bin/env python3
"""Stop only a QuadSmash process recorded by a project run directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quadsmash.process_identity import checked_pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--save-and-stop", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    pid_path = run_dir / "run.pid"
    if not pid_path.is_file():
        print(f"no active pid file: {pid_path}")
        return
    try:
        pid = checked_pid(run_dir)
        if args.save_and_stop:
            (run_dir / "stop.requested").write_text("requested_by=stop_run\n", encoding="utf-8")
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError,FileNotFoundError):
        print("no verified live project process; no signal sent")
    else:
        print(f"sent SIGTERM to project process {pid}; checkpoint handler will save before exit")


if __name__ == "__main__":
    main()
