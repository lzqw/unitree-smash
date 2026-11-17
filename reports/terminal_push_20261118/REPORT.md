# Offline full-prefix contact objective

## Change

The previous one-step correction reduced COM/momentum task error but worsened the strike. This implementation instead scores the terminal racket position, normal and contact velocity after **all 29 intervals**. Twelve leg-command amplitudes multiply one smooth temporal basis; the six prescribed arm commands remain exact. The basis is bounded by the declared contact deadline and current predicted support: flight latches it off. No future native state or case-result lookup is used.

One local solve uses a 0.001 rad one-sided finite difference, a maximum 0.03 rad parameter trust region, and a dimensionless 0.1 ridge residual. The original command bounds further restrict each amplitude. The normalized terminal residual uses 35 mm, 0.12 normal-vector distance, and 0.75 m/s as optimization scales. **Final contact admission remains 3 mm**, with the original normal, velocity and ball-task checks. Lower objective cost is not a teacher certificate.

Each complete sequence is independently integrated from the original q/v, measured contact mask and previous final command. Every step retains original physical diagnostics, authored nonfoot geometry, arm tracking and 2e-6 action mapping checks. Initial nonfoot geometry is checked. The baseline's entire q/v trace exactly matches the preserved prior run. There is no change to native physics, Recovery, scoring, frozen policies or SuperMimic.

## Two CPU implementation attempts, neither a native evaluation

The first implementation always perturbed parameters positively. Four calf-direction trials hit original command bounds and were rejected; its best proposal had 575.253 mm contact error. Its source, complete output, command log and exit 0 are preserved under `attempts/positive_direction01/`.

The implementation was then corrected to choose a finite-difference direction within the original command bounds, with fixed parameters omitted rather than granted artificial freedom. It retains the same maximum **14 complete-prefix calls**: baseline, up to twelve derivatives, and one final proposal. It does not sample additional random candidates or tune physical tolerances. The second CPU attempt is independently recorded in this directory; results from the two attempts are not spliced.

## Final CPU result

| Metric | Exact baseline | New sequence |
|---|---:|---:|
| Complete physical path intervals | 29/29 | 29/29 |
| Terminal position error | 584.672 mm | 566.311 mm |
| Terminal normal error | 0.258904 | 0.240932 |
| Terminal velocity error | 4.392869 m/s | 4.355442 m/s |
| Normalized terminal cost | 318.015476 | 299.558239 |
| Conditional reach deficit | 49.180 mm | 51.467 mm |
| Maximum command encode/decode error | 4.44e-16 rad | 4.44e-16 rad |

All fourteen final-attempt calls passed their 29-step path checks. The selected proposal changes leg commands at nodes 0–12, retaining every arm command exactly. Elapsed CPU time is **107.361 s**, process exit **0**. The local response's normalized linearization error at the independently integrated proposal is 0.024270; this is model-local evidence, not PhysX accuracy.

Contact error improves by **18.361 mm**, but the independent conservative arm-reach deficit becomes worse. The task is still far from admission. `FULLBODY_PREDICTION_VALID=false`, terminal contact fails, and no final collision outcome is certified. PhysX reference, native adoption/hit, Recovery success and complete5 are **NOT_RUN**. No qualified reference is exported, no production path is activated, and no 128-environment evaluation or training occurs.

## Verification

`FINAL_TESTS.txt`: **56 passed**, exit **0**. Tests cover complete-prefix rejection, nonfinite baseline rejection, optimizer failure, boundary-aware derivative direction, fixed coordinates, bounded correction, existing physical admission, command ownership and prefix execution gates. The real CPU rollout, rather than the synthetic optimizer tests, provides the model path and terminal evidence above.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_terminal_push_refinement_v1.py tests/test_nonlinear_push_correction_v1.py tests/test_push_off_window_v1.py tests/test_experimental_sphere_admission_v1.py tests/test_joint_prefix_execution_v1.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.terminal_push_refinement_v1 --output /tmp/terminal_push_new_attempt.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.terminal_push_20261118.analyze
```

`RESULTS.json` retains every trial's amplitudes, complete command/state trace or first rejection, and physical diagnostics. `ANALYSIS.json` retains the terminal Jacobian, singular values, reach bounds and command comparisons. `MANIFEST.json` hashes sources and frozen inputs and verifies protected files unchanged.

## Next step

Do not equate moving the old fixed arm trajectory closer to the ball with creating a reachable new strike: the reach deficit remains 51.467 mm at the predicted base pose. Couple the terminal arm endpoint and base support trajectory, using action-conditioned FK and the task-constrained impact search. Any changed arm sequence must be regenerated and fully re-predicted because its inertia affects the base. The present local Jacobian has a very weak direction (smallest normalized singular value about 0.000193); this is evidence about this fixed temporal basis, not a global impossibility proof. Do not compensate by relaxing physical limits or the final contact gate.
