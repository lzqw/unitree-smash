# Dual line runtime audit

## Scope

This runtime layer only protects process identity and lifecycle boundaries. It
does not modify SuperMimic source, checkpoints, replay, assets, or training
parameters. No SuperMimic process was started by this work.

The teacher code is isolated in
`/home/autodrive/LZQW/unitree_smash_teacher_eval` on branch
`codex/teacher-joint-eval-20261111`. The SuperMimic worktree under `/tmp` is
read-only protected for this task.

## Current process evidence

At the prelaunch probe (`2026-11-16T19:56:18+08:00`) there was no matching
`supermimic`, `teacher`, Isaac, or teacher evaluator process. GPU0 reported
RTX 5080, 5% utilization and 1527 MiB used. The user session cgroup is
`/user.slice/user-1000.slice/user@1000.service`; no installed unitree service
was active.

After the teacher service was launched, an unrelated user-terminal process
`PID 2409630` appeared running `train_supermimic_physical_block.py` in a
different terminal cgroup. Codex did not start, signal, or modify it. Its
presence is recorded as external resource competition, not as part of the
teacher run. It later exited with its own `KeyError: 'episode_index'` while
`physical_loop_replay_v1.py` serialized a replay row. The preserved
`failure.json` reports `native_ticks=0` and `actor_updates=0`; this is an
application failure, not evidence of SIGKILL. No restart was performed.

A later external SuperMimic attempt also exited from its own disk guard at
`native_ticks=272`, `actor_updates=17` because free space was below 8 GiB.
Codex did not restart it. These are independent application/resource failures,
not cross-line termination.

Historical evidence from retry12 remains inconclusive: the service exited with
`status=9/KILL` and no traceback or kernel OOM record. `RuntimeMaxSec` and
`MemoryMax` were unlimited. An external SIGKILL sender is still UNKNOWN.

## Isolation verification

`tests/lifecycle_smoke.py` exited 0 and wrote
`runs/lifecycle_smoke/result.json`. It verified distinct PGID/SID and that
stopping the SuperMimic-labeled smoke service left the teacher-labeled smoke
service alive. A second `systemd-run --user` smoke verified separate cgroups
and the same stop-one-preserve-the-other behavior.

The observer is read-only and sends no signals. The supervisor starts each
child in a new session, records identity and exit evidence, and forwards a
normal stop only to its own child process group.

## Operating boundary

The service unit files are templates and are not installed or enabled by this
task. Real Isaac jobs must be launched only after an explicit teacher-line
run manifest is frozen. The SuperMimic unit is intentionally unused.

## Result

`teacher_strategy_only=true`, `supermimic_modified=false`,
`supermimic_started_by_codex=false`, `teacher_eval_exit=0`,
`teacher_eval_completed=true`, `historical_sigkill_root_cause=UNKNOWN`.
