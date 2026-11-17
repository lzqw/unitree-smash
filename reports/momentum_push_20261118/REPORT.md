# Whole-body momentum objective in offline push-off

## Change and evidence

The preceding contact-window experiment raised COM but worsened pitch and terminal contact. This experiment replaces the body-angular-acceleration objective with an optional whole-body angular-momentum-rate objective in the Approach push-off adapter. It includes all 24 generalized velocities and the arm inertia. It does not inject a desired takeoff state, external torque, or fictitious contact force.

The target is the same-state frozen-command nonlinear prediction's average angular momentum rate over one 10 ms interval. The objective uses `Hdot = Ag_angular * acceleration + dAg_angular * velocity`, about COM in world coordinates. The current locked whole-body inertia normalizes the three objective rows to angular-acceleration units; the existing orientation weight is retained. This normalization does not replace the mass matrix in dynamics. The target is a soft objective, not a promise of momentum preservation.

`--preserve-momentum-rate` is offline and opt-in. Default body-angular tasks are unchanged. A shared QP task hook preserves the original default matrix and target; `DEFAULT_REGRESSION.json` compares original source saved from commit 3512a99 against current code, with and without prescribed arm commands. Both solved fixtures have exactly zero differences in acceleration, torque, force, command, and next joint state. This is bounded regression evidence, not exhaustive equivalence over all states. Recovery code and production activation are unchanged.

## Fixed-input results

| Quantity | Prior contact window | Momentum objective |
|---|---:|---:|
| Path intervals passing original checks | 29/29 | 29/29 |
| Commands adopted within CPU prediction | 10 | 11 |
| Terminal contact position error | 624.335 mm | 584.672 mm |
| Terminal normal vector error | 0.302974 | 0.258904 |
| Terminal velocity error | 4.539563 m/s | 4.392869 m/s |
| Terminal pitch | -0.250805 rad | -0.212901 rad |
| Terminal angular momentum Y | -3.074300 kg m²/s | -2.891231 kg m²/s |
| Conditional reach deficit | 47.100 mm | 49.180 mm |
| First all-foot release | 0.13 s | 0.13 s |

The earlier fixed-eight-tick variant had 598.398 mm contact error. This candidate improves that error by 13.726 mm, but remains far outside final 3 mm contact, 0.12 normal and 0.75 m/s velocity gates. Its reach deficit is worse than the preceding contact-window variant. The arm reference remains fixed in this paired experiment, so this is not a newly optimized complete strike.

The entire predicted command sequence and state trace are in `RESULTS.json`. Maximum original float64 command encode/decode discrepancy is 4.44e-16 rad, below 2e-6; this is not native actuator evidence. CPU rollout elapsed 13.067 s, exit 0. No fullbody-qualified reference, PhysX adoption or hit was produced. Final impact calculation was skipped because terminal gates failed. `FULLBODY_PREDICTION_VALID=false`; PhysX, Recovery success and complete5 are NOT_RUN. No 128-environment run or training occurred.

## Model discrepancy remains material

`MOMENTUM_AUDIT.json` separates requested rate, QP rate, and nonlinear interval-average rate for each adopted command. Maximum QP/nonlinear vector discrepancy is **11.422 Nm**. The QP still uses measured foot-body fixed-contact kinematics and an unsaturated implicit drive map; the nonlinear validator uses authored sphere contact, releases, saturation, and finer integration. Consequently QP task success cannot establish accurate trajectory prediction. All adopted commands were independently re-predicted and checked, but terminal motion is still model-only evidence.

## Tests and reproduction

`FINAL_TESTS.txt`: **31 passed**, exit **0**, including moving-arm centroidal derivative finite-difference verification, physical/drive residuals, task-state cleanup, phase ownership, reset isolation and existing MPC horizon replay tests. The derivative test compares the angular momentum rate against central differences of the loaded fullbody model, not a synthetic three-axis body model.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_push_off_support_qp_v1.py tests/test_implicit_command_support_qp_v1.py tests/test_push_off_window_v1.py tests/test_admitted_arm_support_qp_v1.py tests/test_action_conditioned_base_v1.py tests/test_contact_reach_bound_v1.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.momentum_push_20261118.check_default
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.rollout_push_off_prefix_v1 --contact-window --preserve-momentum-rate --output /tmp/momentum_push_new_attempt.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.momentum_push_20261118.analyze
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.momentum_push_20261118.audit_momentum
```

The recorded rollout output was this folder's `RESULTS.json`; existing attempts cannot be overwritten. Source diff, hashes, command exits, prior QP source and raw results are preserved. Protected controller, scoring, checkpoint and input hashes are unchanged.

## Next action

Before choosing larger push-off requests, reconcile the QP contact-point/drive assumptions with the independent nonlinear validator. The 11.422 Nm discrepancy is direct evidence that desired angular momentum is not yet reliably realized. Then use the action-conditioned base to regenerate the arm reference and repeat strict complete-prefix and ball-task validation. A remaining 49.180 mm conservative reach deficit calls for executable base/timing changes; it is not a license to relax joint, clearance or contact limits. Keep this option off by default and do not launch a native strike candidate yet.
