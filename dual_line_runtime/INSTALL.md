# Installation and run contract

Install the four unit files into `~/.config/systemd/user/` only after checking the line-specific worktree and command JSON. Keep each run under a fresh directory:

- `/home/autodrive/LZQW/dual_line_runtime/runs/supermimic/<run_id>/`
- `/home/autodrive/LZQW/dual_line_runtime/runs/teacher/<run_id>/`

Each run directory must contain `manifest.json`, `command.json`, `identity.json`, `stdout.log`, observer JSONL, and `exit.json` or an explicit interruption receipt. The two command JSON files must reference different worktrees and different writable output/checkpoint roots. Read-only assets may be shared only with a recorded SHA256.

The service units use `Restart=no`, `RuntimeMaxSec=infinity`, `TimeoutStopSec=infinity`, and separate slices. External scheduler limits remain authoritative; the units do not bypass them.
