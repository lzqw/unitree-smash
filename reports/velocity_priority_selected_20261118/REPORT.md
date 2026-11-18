# Velocity-priority whole-prefix refinement

This is a matched re-optimization from the already selected projected-goal
prefix. The predecessor's complete q/v state, 18D final command history and
planned arm references are restored and checked node by node. The analytic goal
was explicitly verified for the same contact event and causal ball before this
run. Target metadata differs from the earlier goal, so normalized residuals are
not compared across goals; the physical full-body state/history is the baseline
identity check.

The optimizer objective is racket-velocity residual only. Position and normal
terminal groups remain hard non-regression constraints against the selected
predecessor under the current projected goal. Original arm tracking, command
limits, path constraints, dynamics and mapping checks remain active. No
fixed-face speed sweep, parameter scan, native run or training was performed.

## Matched result

| physical residual | predecessor selected | velocity-priority selected |
|---|---:|---:|
| contact position | 111.588458 mm | **110.951216 mm** |
| normal vector | 0.187343 | **0.123051** |
| racket velocity | 2.775041653 m/s | **1.868392362 m/s** |

The selected proposal is a real 18D command change, not a goal rename. It
passed all 33 prediction intervals. One full proposal failed at node 16 with
arm tracking 0.250141443 rad and was rejected. The accepted path's maximum arm
tracking error and command round-trip error are in `MATCHED_AUDIT.json` and are
checked against the unchanged limits. Runtime 229.4576 s; 27 fresh rollouts;
replay exit 0. Matched baseline/history audit exit 0.

The complete hitting gates still fail: position 110.95 mm > 3 mm, normal
0.123051 > 0.12, velocity 1.8684 m/s > 0.75 m/s. `FULLBODY_PREDICTION_VALID`
therefore remains false. This is meaningful reduction of the speed deficit,
but not a valid teacher reference or a native-hit result.

## Verification

46 CPU regression tests passed, exit 0. Protected physics, scoring, Recovery,
frozen base, weights and input assets were unchanged. No future native state was
used online; CPU future state is diagnostic model output only.

`PHYSX_REFERENCE_VALID=NOT_RUN`, `NATIVE_HIT=NOT_RUN`,
`RECOVERY_SUCCESS=NOT_RUN`, `COMPLETE5=NOT_RUN`.
`evaluation_stage=teacher_debug`, `formal_eligible=false`,
`formal_acceptance=false`, `formal_cases_read=false`.

Next modification should address the remaining ~111 mm contact-position error
and the last normal/velocity tracking gap with a true change of contact event,
base state or coupled task geometry. Do not relax the 3 mm, 0.12, 0.75 gates or
claim native improvement from this CPU model.
