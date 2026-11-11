# Direct command-sequence rescue evaluation

Date: 2026-11-11 (Asia/Shanghai)

## Status

`NATIVE_EVALUATION_ABORTED_BEFORE_CONTROL_TICK`.

The single authorized native launch initialized Isaac Sim and the 128-environment scene, then stopped before the first controller tick because the copied evaluator imported the new adapter from the old repair module. The process exit code was 0, but `native/completed.json` was not created, so this run has **no valid task metrics** and must not be treated as a 128-case result.

Error recorded in `native/error.txt`:

```text
ImportError: cannot import name 'NativePureControlStateDrivenLongMPCV1QPContinuityDirectSearch'
from scripts.native_pure_control_state_driven_long_mpc_v1_qp_continuity_repair
```

The evaluator import and adapter were corrected after the run and pass a Python import check. No second native run was started, in accordance with the one-run instruction.

## CPU evidence before the aborted run

- CPU gate: `CPU_PREFLIGHT_PASS`.
- Fixed recovery set: 74 cases.
- Direct candidates: two state-derived full-authority 32-node sequences (leg extension factors 2.25 and 3.0).
- Full candidates accepted without execution: TRAIN 7, 17, 23, 32, 37, 60, 68 (7/74).
- No partial candidate execution; original limiter preview and branch-local final-command history were used.
- Saved fallback replay: 2,759 prediction nodes, 1,424 solver records; geometry conflicts were concrete `ground / Head_lower` (82) and four `authored_net / Racket::net` events, with no foot-pair geometry conflicts. Solver records included 1,368 solved and 56 primal-infeasible nodes. No concrete checker implementation error was found in the CPU replay.

## Metrics

The requested candidate columns are unavailable because the native run did not reach a control tick. The saved work comparator remains the valid reference:

| Metric | Historical 52/74 baseline | Work comparator 55/74 (`7251b5f`) | Direct candidate |
|---|---:|---:|---:|
| Qualified aerial hit | available in saved baseline reports | 74/128 | **not evaluated** |
| Ball-side conditions except recovery | available in saved baseline reports | 46/128 | **not evaluated** |
| Conditional recovery | 52/74 | 55/74 | **not evaluated** |
| Complete5 | 39/128 | 39/128 | **not evaluated** |
| Post-takeover head-free and recovered | 0/74 | 0/74 | **not evaluated** |
| Post-takeover non-foot-free and recovered | 0/74 | 0/74 | **not evaluated** |
| Numerical faults / command audit | 0; audit passed | 0; audit passed | **not evaluated** |

No retention/loss/new set can be computed for the candidate, and no physical head-contact conclusion can be drawn from this aborted run. The prior CPU fallback replay indicates that head-ground geometry is a dominant rejection object before persistent fallback, but it is not a result of the direct-search native evaluation.
