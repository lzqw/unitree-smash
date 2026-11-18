# Separate terminal-task protection in full-prefix joint refinement

Base commit: 3c10920198d5c60d781bca94c553c317740a5ee1. Offline teacher development only. No physical-limit, asset, scorer, frozen base, Recovery or SuperMimic changes; no training or native/128-case evaluation.

## Problem and implementation

The previous weighted terminal cost selected a path with position error 378.014 mm but normal error increased from 0.141881 to 0.186709 and velocity error also increased. That candidate was correctly not a qualified teacher, but the local improvement label masked task tradeoffs.

The opt-in `--protect-terminal-tasks` keeps position, normal and velocity as three independent inequalities in the local response optimization. Each squared residual-group norm is constrained to be at most its baseline value, with analytic constraint Jacobians. Original combined command constraints and bounded parameters remain active. After the local solve, the independently recomputed full nonlinear prefix must satisfy all three nonregression tests as well as total-cost reduction. Normalized comparison tolerance is 1e-9; this is numerical comparison slack, not a physical/task acceptance threshold. Final contact and ball-side task gates remain unchanged.

Default optimizer behavior is preserved. Backtracking is still the fixed 1, 1/2, 1/4 schedule, stopping at the first complete verified improvement. Every candidate starts from the same true initial state, actual previous command and restored reference history. It never continues from a failed partial prefix.

## Same-input CPU result

Input remains the previous 425.248711 mm coupled prefix, **not** the regressed-normal 378 mm diagnostic candidate. All source/input/protected hashes checked after completion. 27 fresh model calls; no response-cache reuse.

| Terminal quantity | Baseline | Protected half-step |
|---|---:|---:|
| Position error | 425.248711 mm | 385.969234 mm |
| Normal-vector error | 0.141881005 | 0.119489675 |
| Requested racket velocity error | 4.186762787 m/s | 4.164683755 m/s |

All three errors decrease on the same complete predicted sequence. Normal passes 0.12; position and velocity fail 3 mm and 0.75 m/s. This is not submillimeter IK precision, a native collision result, or a qualified teacher. Relative to the previous unconstrained half-step, position is worse (385.97 vs 378.01 mm) in exchange for enforcing task nonregression; do not claim improvement against every historical metric.

The full-step proposal failed at zero-based node15 with arm tracking 0.257425750732 rad > 0.25. The half-step independently passed all29 intervals. Maximum arm tracking: 0.246936892451 rad; CPU command-mapping round trip: 4.4408920985006262e-16 rad. No failed prefix is executed. Raw rejected/accepted trajectories and reference histories are retained in RESULTS.json.

`FULLBODY_PREDICTION_VALID=false` because final task/contact fails despite path-level validity. Forward impact is NOT_EVALUATED; native adoption/hit, PhysX reference, Recovery success and complete5 are NOT_RUN. No reference is installed/exported as qualified. The experimental model is not claimed equivalent to PhysX.

## Tests and timing

87 tests passed, exit0 (`TESTS_FINAL.txt`). New tests demonstrate: weighted-cost normal regression is prevented; nonlinear rollout regression is rejected even if the local model passed; simultaneous task improvements remain possible; no success is promoted to a native/teacher certificate. Existing full-prefix, reference-isolation, QP, support-window, action-bridge and bundle checks pass.

CPU replay took 203.910252s, exit0. Offline generation only; not realtime. Exact replay command is in COMMANDS.txt, dirty patch in SOURCE.diff, and hashes in MANIFEST.json. No thresholds were relaxed.

## Remaining work

The next limiting tasks are predicted contact position (386 mm) and velocity (4.165 m/s tracking error), not further normal-only refinement. Use the same fully propagated terminal state to diagnose reachability and base/arm timing. Preserve the current three-task protection during subsequent changes. A bounded local model cannot prove that no feasible joint trajectory exists. Do not increase fixed-normal rate trials or start a 128-case evaluation. Independent PhysX calibration and eventual full task/native closure remain outstanding.
