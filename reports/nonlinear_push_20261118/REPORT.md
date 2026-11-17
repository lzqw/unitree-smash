# Fine-step nonlinear command correction: local improvement, terminal regression

## Implementation

Added `nonlinear_push_correction_v1.py`, an offline-only response correction for twelve leg position targets. The six arm commands remain exactly unchanged. Each correction starts from the current state and previous actual command, and independently runs the existing 160-substep saturated-drive/contact model. There are twelve bounded one-sided derivative evaluations, one regularized bounded least-squares solve, and one independently integrated proposal. Maximum calls are 14 per correction including its baseline. There is no random sampling, parameter scan, force injection or change to physical parameters.

The finite-difference step is 0.001 rad; the local command trust radius is 0.03 rad intersected with the original position limits. This radius is an optimization restriction, not an actuator-limit change or an extra runtime slew limiter. The dimensionless ridge coefficient is 0.1 in the least-squares residual. The task residual is interval-average COM acceleration normalized by 9.81 m/s² and COM angular momentum rate normalized by `mass * 9.81 * 0.1 m`. Desired values retain the preceding experiment's same-state nominal response plus requested COM acceleration increment, and nominal angular momentum rate.

Physical checks are separate from cost. Invalid baseline, invalid derivative or rejected final proposal cannot become an execution certificate. Rejected correction returns the unchanged QP command; the outer rollout still independently checks it. The adopted proposal is integrated again before the rollout advances. Nonfoot contact, geometry, torque, speed, original arm tracking, command history and 2e-6 mapping checks remain in force. A lower local cost does not waive complete-prefix or terminal task admission.

The explicit CLI flag is `--nonlinear-push-correction`, requiring `--preserve-momentum-rate`. It is off by default and is not attached to the production controller or Recovery. No SuperMimic, learned policy, asset or scoring changes occurred.

## Actual fixed-input CPU result

- 10 correction calls, 140 model evaluations, no failed physical evaluations; all 10 new commands were adopted **within this CPU rollout**.
- Every adopted correction lowers its own same-state nonlinear task cost. Sum of those paired local costs decreases from **9.257543 to 7.511933**, or **18.86%**. These are local response costs, not an episode score or success rate.
- Maximum normalized discrepancy between the local linear response prediction and its independently integrated proposal is **0.037263**. This validates local consistency with the CPU model only, not accuracy against PhysX.
- All 29 path intervals pass, but terminal position error increases from **584.672 to 596.859 mm**, normal error from **0.258904 to 0.269786**, and velocity error from **4.392869 to 4.422782 m/s**.
- Conditional reach deficit grows from **49.180 to 53.286 mm**. First all-foot release stays at 0.13 s. Maximum command round-trip error is **4.44e-16 rad**.
- CPU elapsed **63.771 s**, exit **0**. This is an offline calculation and is not suitable for a synchronous 10 ms control loop.

Thus improved one-interval tracking of this COM/momentum request does not improve the final strike. The simple desired takeoff task remains misaligned with the contact objective; the old arm reference is also unchanged. No candidate is promoted, exported as qualified, or sent to native evaluation. `FULLBODY_PREDICTION_VALID=false`; final impact is `NOT_RUN_TERMINAL_REJECTED`. PhysX reference, native adoption/hit, Recovery success and complete5 remain NOT_RUN. No 128-case evaluation or training was run.

## Verification

`FINAL_TESTS.txt`: **62 passed**, exit **0**. Tests cover independent leg directions, fixed arm commands, bounded proposals, failed physical admission, nonfinite baseline rejection, history isolation, existing implicit MPC checks, phase ownership, and final-prefix execution gates. `CORRECTION_AUDIT.json` verifies actual saved commands match accepted proposals, and all arm commands match the original QP reference. Original physical diagnostics and final state sequence are retained in `RESULTS.json`.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_nonlinear_push_correction_v1.py tests/test_push_off_support_qp_v1.py tests/test_implicit_command_support_qp_v1.py tests/test_push_off_window_v1.py tests/test_experimental_sphere_admission_v1.py tests/test_joint_prefix_execution_v1.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.rollout_push_off_prefix_v1 --contact-window --preserve-momentum-rate --nonlinear-push-correction --output /tmp/nonlinear_push_new_attempt.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.nonlinear_push_20261118.analyze
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.nonlinear_push_20261118.audit_corrections
```

The recorded output is this directory's `RESULTS.json`; existing attempts cannot be overwritten. Hashes and protected-input verification are in `MANIFEST.json`.

## Next step

Use complete-prefix contact/reach consequences to define the offline push-off correction objective, rather than continuing to optimize the current hand-derived one-step COM/momentum request. Recompute the arm endpoint against the resulting action-conditioned base and revalidate every changed sequence. Preserve failure evidence and do not increase gains, trials, or physical tolerances to hide the terminal regression. Accurate local response is useful infrastructure, but is insufficient for a valid fullbody strike.
