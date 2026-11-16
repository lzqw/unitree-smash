# Recovery Arm WBC Next Round Report

## Status

- `evaluation_stage = teacher_debug`
- `formal_eligible = false`
- `formal_acceptance = false`
- `formal_cases_read = false`
- **One valid native teacher evaluation completed; execution stops here.**

## Results (128 environments)

| Metric | Current | Historical |
|---|---:|---:|
| Qualified airborne hit | 28/128 | 29/128 |
| Conditional recovery | 19/28 | 20/29 |
| complete5 | 6/128 | 10/128 |
| Actual racket contacts | 59 | 59 |
| Crossed net | 26 | 27 |
| Landed target | 18 | 21 |
| Numerical failures | 0 | 0 |

The conditional recovery rate is 67.86% (19/28), versus 68.97% (20/29). The result does not exceed the historical baseline and is a regression in all three requested primary counts.

## Paired case comparison

On the saved 128-case panel, airborne hits retained 25, lost 4 (`1,11,51,58`), and gained 3 (`22,115,118`). The raw `recovery_success` episode flag retained 16, lost 5, and gained 5; the official metrics field is the stricter 19/28 conditional count and is the value used above. complete5 has no raw boolean field in the episode records; the official metric is 6/128 versus 10/128.

Full lists are in `COMPARISON_29_20_10.json`.

## MPC and arm evidence

The direct-search fix was exercised in native execution. There were 897 trigger calls, 7176 candidate samples, 8 full-feasible candidates, and 2 adopted complete 32-node sequences. Rejections were 7018 physical violations and 158 prediction failures. Thus the improvement was actually adopted in native execution, not merely found in CPU search. The legacy `pure_control_recovery.sequence_mpc=false` field was not updated by this path; the per-tick direct-search diagnostics are authoritative.

Measured contact-point speed over 59 saved racket-contact events was max 3.011 m/s, median 1.053, p95 1.885; historical max was 2.689 m/s. The candidate raised the peak but did not improve task success. The command-target step diagnostic reached 58.48 rad/s when divided by 0.01 s; this is not a measured physical joint velocity and does not override native actuator limits.

## Command and teacher isolation

All 28,687 saved command rows had an execution preview. Prediction preview versus native arbiter was 0 rad; normalized dispatch round-trip was `1.787e-7 rad`; actual-history continuity was `3.619e-7 rad`; all are below `2e-6`. Requested-to-final differences were zero in this full-authority path. Actor forward count and optimizer steps were both zero, and all three learned action contributions were disabled.

The native service exited code 0 with no signal and no stop request. Attempts 01 and 02 are preserved as engineering failures and were not counted as results. SuperMimic was not modified or stopped by this work; its later absence from the process table is recorded as UNKNOWN external termination because no direct termination evidence was obtained.

## Answer to the requested questions

1. Yes, one valid native teacher evaluation completed.
2. No: 19/28 conditional recovery and 28/128 airborne hits do not exceed the historical 20/29 and 29/128; complete5 is 6/128, below 10/128.
3. Airborne hits: retained 25, lost 4, new 3. Raw recovery flags: retained 16, lost 5, new 5. Official conditional recovery count is 19 versus 20.
4. Hit and complete5 regressed; conditional recovery rate also slightly regressed.
5. No net clean-recovery improvement was demonstrated; no complete5 improvement.
6. The direct-search sequence was actually adopted twice in native execution.
7. Exit state, numerical failures, command round-trip, history continuity, and `2e-6` checks were verified; frozen candidate manifest is saved.
8. Next priority is DTG/Approach contact timing, racket normal, and velocity shaping, followed by bounded direct-search feasibility/objective refinement. See `NEXT_ROUND_RECOMMENDATIONS.md`.
