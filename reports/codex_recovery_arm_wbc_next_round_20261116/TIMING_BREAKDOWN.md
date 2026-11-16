# Timing and Resource Evidence

The native service ran from `2026-11-16T16:44:06.369211+00:00` to `2026-11-16T18:58:27.265552+00:00` (wall time about 2 h 14 m 21 s) and exited normally with code 0. The candidate manifest records the command, cgroup, GPU snapshot, and disk snapshot.

The evaluator did not emit per-stage nanosecond timing for state extraction, DTG reference, WBC, sequence generation, 32-node prediction, geometry checks, execution mapping, logging, or simulator stepping. Those fields are therefore `NOT_RECORDED`; no synthetic p50/p95/p99 values are reported. The saved diagnostics contain per-plan `wall_seconds` and 28,687 command rows, while the raw JSONL is retained for offline timing analysis.

Observed resources at freeze: RTX 5080, 5917 MiB / 16303 MiB used; teacher process was in the independent `unitree-teacher-recovery-arm-wbc-20261116-03.service` cgroup. The SuperMimic process snapshot in the manifest was separate. Final GPU process inspection found no surviving SuperMimic training PID; that later termination is `UNKNOWN`.
