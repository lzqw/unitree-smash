# Teacher debug evaluation audit

This work is on isolated worktree `codex/teacher-joint-eval-20261111` at
`2ab8eb0699d2b5513540792d3b60d7d7df9142d7`, the verified 7251b5f/55-74
execution-consistency line. The active source and unrelated SuperMimic
worktrees were not modified or started.

The current teacher candidate is
`attempts/teacher_joint_eval_20261111_01/CANDIDATE_MANIFEST.json`. Its
algorithmic change is a deterministic full18-joint, 32-node direct search
after normal and repaired Recovery plans both fail. Feasible candidates now
rank head and non-foot clearance as separate normalized terms, in addition to
the existing tilt, momentum, continuity and support terms. Normal plans,
repaired plans, limits, command history, arbiter semantics, and 2e-6 replay
checks remain unchanged.

CPU/static prechecks passed (`17 passed`, py_compile, native-entry static
checks, and all candidate hash checks). The teacher-only entry disables the
residual actor and rejects learned recovery corrections, expert banks and
optimizer paths.

One teacher-only native evaluation was completed in the dedicated service
`teacher_joint_20261116_01`. It ran 800/800 ticks and 128/128 first episodes,
then exited normally with code 0. Its raw metrics and command audit are under
`attempts/teacher_joint_eval_20261111_01/native_teacher_joint_20261116_01/native`.
The formal flags remain false. The direct-search entry was attempted but all
333 attempts failed with `ValueError: Actual command history required`, so no
direct-search sequence was adopted.

The independent runtime smoke at
`/home/autodrive/LZQW/dual_line_runtime/runs/lifecycle_smoke/result.json`
passed: distinct PGID/SID, stopping line A preserved line B. A separate
systemd user-unit smoke also showed line B active after line A was stopped.
