# Full-prefix tracking constraints in joint contact refinement

Offline teacher development from eb1c508fe602e49f1859fcb027fe4e2487e63672. No training, SuperMimic operation, native/128 evaluation, asset/scorer/physical-limit change, or Recovery modification.

## What changed

The previous0.33s refinement rejected full, half and quarter steps for arm tracking above0.25rad. The local optimizer constrained terminal tasks and command positions but lacked intermediate tracking constraints.

New `arm_tracking_response` extracts signed actual-minus-planned arm positions for every node and all six actual mapped joints. It rejects incomplete, unordered, nonfinite, inconsistent or falsely labelled verified histories. The original0.25rad value is now a shared constant for the unchanged nonlinear gate and local-model bounds. No physical tolerance is relaxed.

The optional path-response callback estimates a33×6 response Jacobian from the same full-prefix derivatives. SLSQP includes all198 two-sided inequalities alongside original command bounds and separate terminal-task nonregression. Each final proposal is independently fully integrated; actual responses are rechecked against the original bounds. The linear solution, a failed prefix, or an averaged sequence is never an execution certificate. Defaults without the optional callback remain unchanged.

The retimed metadata now nests old-event IK/geometry/velocity diagnostics under `seed_metadata`; old source claims no longer sit beside the current0.33s target. Newly computed forward ball-task results remain separate from robot feasibility.

## Same-input result

| Metric | Baseline | Verified half-step |
|---|---:|---:|
| Position error | 202.401244mm | 173.850565mm |
| Normal-vector error | 0.334498436 | 0.330989113 |
| Racket velocity tracking error | 2.999898897m/s | 2.990948934m/s |

All three terminal residuals decrease against the same0.33s goal. All33 path intervals pass for the half-step. Nevertheless, final3mm/0.12/0.75m/s gates all fail. The analytic target exists, but the actual robot contact/output does not qualify. `FULLBODY_PREDICTION_VALID=false`, no qualified teacher is exported or installed, and PhysX reference/native hit/Recovery/complete5 remain NOT_RUN.

The full step was still rejected at zero-based node16 with0.250093503rad tracking error. It was not rounded down or admitted. Half-step maximum actual tracking error is0.248491930rad. Maximum difference between local linear tracking prediction and independently integrated response is4.5881566659e-05rad, at zero-based node31, arm index0. This compares two CPU computations, **not** model-vs-PhysX accuracy. The maximum CPU command mapping error is4.4408920985006262e-16rad; native execution2e-6 audit remains NOT_RUN.

## Validation

58 tests passed, exit0. New tests cover per-node signed mapping, copies, missing/nonfinite/incorrect history, fixed original bound, step shaping by path constraints and nonlinear rejection despite a locally feasible prediction. Old-event claims cannot leak into retimed target acceptance. Existing terminal, reference-history, contact geometry, bundle and execution-interface tests pass. An initial isolation test incorrectly compared a rounded synthetic0.20000000000000004 to literal0.2; the fixture was corrected to compare the exact pre-mutation array. Its failed output is retained. No production threshold changed to pass tests.

CPU run:27 fresh full-prefix calls, 218.636144s, exit0. Initial state, original final command, model inputs and protected files were checked. Baseline q/v exactly reproduces the prior33-node seed. Source and input hashes were verified after the run. Raw complete and rejected trials, command/reference histories, predicted and actual tracking vectors, logs, commands and dirty diff are retained.

## Next evidence-based work

Intermediate tracking is now represented in the optimizer and no longer rejects all bounded steps. Remaining terminal errors are still large. Continue only with complete-sequence causal response/retiming and joint base/arm targets, preserving original path and terminal gates. Keep the distinction between this model's local response accuracy and native predictive accuracy; independent PhysX calibration and actual task closure remain unproven. No128-case run is warranted.
