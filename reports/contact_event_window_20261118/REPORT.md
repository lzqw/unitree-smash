# Original contact-window causal full-prefix extension

This offline development change investigates the remaining reach/timing mismatch. No native evaluation, training, SuperMimic modification, physical/scoring change, or Recovery change. No future native states are read.

## Implementation

The source causal search already allowed contact times 0.29, 0.31 and 0.33 s. The new script obtains these times from the original event record, not from success lists or a new parameter scan. It starts from the same true initial state and previous final command. It preserves the selected29-command prefix and its planned reference history, then explicitly holds the last18D command and last reference for four extra ticks. This simple causal hold is only a diagnostic extension, not an optimized strike or takeoff action.

The complete33-tick sequence is freshly integrated at160 substeps/tick with original contact, position, velocity, torque, tracking and clearance checks. The first29 predicted q/v states are exactly equal to the saved source. No root extrapolation, translation injection, fabricated support force or partial-prefix certificate is used.

Ball positions and incoming velocities at all three times are propagated from the single originating observation at tick249. For each event, the analytic task goal is recomputed through the forward impact model. Conditional IK and velocity checks use that event's fully propagated root state. Counterfactual arm poses are never installed and must undergo new fullbody validation if used to form a new trajectory.

## Results

| Contact time | Predicted actual contact error | Position-only local IK | Conservative reach deficit | Analytic task goal |
|---|---:|---:|---:|---|
| 0.29s | 385.969mm | 206.657mm | 49.449mm | found |
| 0.31s | 281.703mm | 112.083mm | 0 | found |
| 0.33s | 202.401mm | 30.358mm | 0 | found |

Zero conservative reach deficit is only a necessary geometric condition, not proof of reachable contact. At0.33s the position-only IK enters the original35mm coarse envelope, but fails the3mm final gate and has normal-vector error1.523. Consequently the bounded rate solve is correctly NOT_SOLVED; its reported residual must not be interpreted as an optimized velocity. All three actual forward collision checks reject ball/string position mismatch. No native collision or valid teacher reference exists.

The analytic task margins decrease for the later events (minimum normalized margin approximately0.704,0.399,0.146). Ball heights are1.8645,1.7975,1.7269m respectively: delaying contact improves reach here while reducing return/clearance margin. This is a concrete tradeoff to retain in joint timing/base/arm optimization, not permission to keep delaying until contact becomes convenient.

33/33 model path intervals pass. The final diagnostic run took9.270s and exited0. Initial path-only run RESULTS.json (9.265s, exit0) is preserved separately. Both runs logged SciPy's bounded-optimizer trial clipping warning; no physical command/state was clipped or physical bound enlarged. The forward model independently accepted the analytic goals, but they remain unverified robot targets.

## Tests and provenance

41 tests passed, exit0 (TESTS_FINAL.txt). New extension tests verify original command/reference preservation, copied storage, nonfinite rejection, strict four-tick extension bound and no truncated histories. Existing loaded-FK, coordinate/Jacobian, geometry, anchored pose and full-prefix refinement tests pass. Source/input/protected hashes and exact commands are saved. After the diagnostic run, the observation-clock assertion was replaced with explicit ValueError to retain the guard under Python -O; tests reran. SOURCE_EXECUTED.py preserves the actual executed source and the manifest records this guard-only difference.

## Next step

Retarget the arm and leg/base trajectory jointly at the original0.33s event, beginning from actual state/history and preserving task nonregression. The conditional position gap is now30mm rather than207mm, but fixing it while achieving the newly valid task normal requires a new full18D sequence, not substituting IK output or teleporting the base. Maintain original3mm,0.12 and physical gates; do not promote this hold tail as a teacher and do not start128-case/native candidate evaluation.
