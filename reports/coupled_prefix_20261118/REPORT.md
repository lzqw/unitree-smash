# Coupled leg and arm full-prefix refinement

## Implementation

Added `CoupledContactRollout` over the independently validated full-prefix evaluator. Its 24 continuous parameters are twelve support-leg amplitudes, six arm endpoint offsets, and six arm endpoint-rate basis amplitudes. All parameters are radians; endpoint rate offsets are the last six divided by the 0.29 s horizon. Arm correction uses `f(s)=3s²−2s³` and `g(s)=s³−s²`, so initial position and velocity corrections are zero and terminal position/rate can change independently.

Leg corrections stop when predicted support ends; arm commands may continue in flight. The loaded 24-DOF dynamics include their reaction on the base. No ideal base state, ground force or body torque is injected. Original actuator, joint-speed, contact, nonfoot clearance, tracking and 2e-6 command checks remain unchanged. Final contact, normal, velocity and ball-task gates remain independent of optimizer cost.

Fixed local parameter radii are 0.03 rad for legs, 0.1 rad for arm endpoint offsets, and 0.05 rad for arm rate-basis amplitudes; these are optimizer trust regions, not hardware limits. A 0.001 rad finite difference is used, with direction selected inside original bounds. There is one local solve and at most 26 response evaluations; no random sampling or parameter sweep is introduced. The production controller, Recovery, scorer, assets, frozen policy and SuperMimic are unchanged.

## Rejected implementation attempt and correction

`attempts/box_only01/` preserves the first CPU attempt, sources, log and exit 0. Its final combination violated an original joint target limit at node 18 and was rejected. Individually feasible endpoint-position and endpoint-rate changes are not necessarily feasible when added.

The final implementation therefore places their **combined** linear command/reference inequalities into the optimizer: 348 original arm-bound rows over 29 nodes. It solves this constrained quadratic local problem using SLSQP, then independently evaluates the new complete sequence. It does not clip the proposed command to manufacture acceptance.

To avoid repeating identical expensive model calculations, the baseline and 24 derivative responses were reused only after checking input hashes, rollout-method source identity, model/source/configuration signatures, interpreter and library versions. The signature for the first attempt was captured retrospectively, with method ASTs checked against its archived source; `CACHE_PROVENANCE.json` records that fact. This is not a claim that the first attempt had a pre-run signed manifest. The optimizer changed, but the response evaluator did not. The **new combined proposal was not cached**: all 29 intervals were freshly integrated from the initial state and previous actual command. No partial trajectories were stitched.

## CPU results

| Metric | Baseline | New coupled sequence |
|---|---:|---:|
| Complete model path intervals | 29/29 | 29/29 |
| Terminal position error | 566.311 mm | 425.249 mm |
| Terminal normal vector error | 0.240932 | 0.141881 |
| Terminal velocity tracking error | 4.355442 m/s | 4.186763 m/s |
| Normalized terminal cost | 299.558239 | 180.182173 |
| Conditional reach deficit | 51.467 mm | 50.935 mm |
| Maximum command mapping discrepancy | 4.44e-16 rad | 4.44e-16 rad |

The predicted racket speed magnitude **at the planned event time** decreases from 2.3493 to 1.8557 m/s, even though its vector tracking error decreases. These are model velocities at a missed contact, not measured native hit speeds. The vector values and joint velocities are in `ANALYSIS.json`; no speed increase is claimed.

The coupled sequence changes arm commands as well as support-leg commands, and changes the predicted base orientation. Terminal position improves by **141.063 mm**, but all final strike requirements are still unmet. The target remains at least 50.935 mm outside the conservative arm reach bound at this predicted base pose. That bound is conditional on time/base pose and is not an executable takeoff correction.

`FULLBODY_PREDICTION_VALID=false`, terminal contact fails, and final impact is not evaluated. PhysX reference, native adoption/hit, Recovery success and complete5 are **NOT_RUN**. No reference is promoted or exported as qualified, no training occurs, and no 128-environment evaluation is started.

## Timing and tests

The first full response calculation took **190.865 s**. After adding combined constraints, reusing 25 identical responses and freshly integrating the final proposal took **7.272 s**. This is not a 7.272 s cold-planning benchmark and is not a real-time controller.

`FINAL_TESTS.txt`: **63 passed**, exit **0**. Coverage includes endpoint continuity and independent position/rate effects, fullbody parameter participation, original combined command inequalities, boundary-aware derivatives, exact-cache invalidation on changed rollout semantics, physical admission and prefix execution gates. Both CPU attempts exited 0; controller acceptance remains separate from process success.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_coupled_contact_prefix_v1.py tests/test_terminal_push_refinement_v1.py tests/test_nonlinear_push_correction_v1.py tests/test_push_off_window_v1.py tests/test_experimental_sphere_admission_v1.py tests/test_joint_prefix_execution_v1.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.coupled_contact_prefix_v1 --output /tmp/coupled_prefix_fresh.json
# Optional exact-response reuse; a new final proposal still gets full integration:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.coupled_contact_prefix_v1 --reuse-responses reports/coupled_prefix_20261118/attempts/box_only01/RESULTS.json --output /tmp/coupled_prefix_reuse.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.coupled_prefix_20261118.analyze
```

`RESULTS.json` retains complete command/state sequences, residuals, bounds, costs and rejection evidence. `MANIFEST.json` records protected-file verification and input/source hashes; `RESULTS.evaluation_signature.json` records exact-response cache scope. Result-integrity sidecars were added after the first reuse to protect subsequent reuse; this metadata change does not alter the recorded physical computation. Sidecars are named per output file to avoid overwriting another attempt’s metadata.

## Remaining work

The coupled sequence is a substantially better model candidate, not a hit. Re-linearization must use its new action-conditioned state trajectory and exact command history. Further optimization must also address the remaining base/timing reach deficit rather than only rotating the arm. Every changed sequence needs fresh complete validation; the current result cannot certify a future combined sequence. Keep the 3 mm, normal, velocity and ball-task gates, and do not start native evaluation yet.
