# Dual line runtime isolation

This directory owns service lifecycle only. It does not share replay, weights, locks, or writable run directories between SuperMimic and the DTG/WBC/Recovery teacher.

Each line runs in its own user systemd slice and service. `line_supervisor.py` creates a new session for the actual Isaac process, records PID/PPID/PGID/SID/cgroup/start ticks, and writes `exit.json`. `line_observer.py` is read-only and writes JSONL telemetry; it never signals a target.

The service units are templates. Before starting a real run, create a per-run command JSON, run manifest, and environment file or instantiate the unit with `systemd-run --user` and explicit properties. Do not use a Codex shell as the lifetime owner.
