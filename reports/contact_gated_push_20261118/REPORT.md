# Causal support-window experiment

This is a CPU-only continuation of the task-constrained teacher work. It does not establish a valid teacher reference. The new option is not promoted: terminal contact regressed.

## Implementation and boundaries

`PushOffWindowV1` uses only the current predicted four-foot contact mask, an explicit contact deadline, and ownership. Flight, deadline, and Recovery handoff latch the window off; later contact cannot reactivate it. Independent instances/reset isolate lanes. The offline rollout exposes `--contact-window`; the default remains the previous eight-tick window. No production controller, Recovery, physical limits, scoring, or SuperMimic changes are included.

Each adopted QP command still undergoes the independent nonlinear prediction and original path checks. Completing path checks does not waive terminal position, normal, velocity, or ball-task checks. The old arm reference is retained in this experiment; this is not a jointly reoptimized arm trajectory or proof of global infeasibility.

## Same-input CPU comparison

| Metric | Fixed eight ticks | Current-contact window |
|---|---:|---:|
| Complete path intervals | 29/29 | 29/29 |
| QP commands adopted in prediction | 7 | 10 |
| Terminal position error | 598.398 mm | 624.335 mm |
| Terminal normal vector error | 0.261512 | 0.302974 |
| Terminal velocity error | 4.431020 m/s | 4.539563 m/s |
| Terminal COM height | 0.569310 m | 0.579832 m |
| First all-foot release | 0.13 s | 0.13 s |
| Conditional reach translation lower bound | 57.879 mm | 47.100 mm |
| Maximum command round-trip error | 4.44e-16 rad | 4.44e-16 rad |

The new window requests QP at nodes 0–12, adopts ten commands, and latches off at node 13. There are no airborne QP calls. The empty `enabled_nodes` array for the historical comparison means that its older log lacks window metadata, not that it made no QP calls.

Higher COM does not imply a better strike: terminal pitch changes from -0.210598 to -0.250805 rad and centroidal angular momentum Y from -2.782843 to -3.074300 kg m²/s. These coupled changes accompany worse terminal contact. The reach bound improves but still excludes the target at the predicted base pose. It is a conditional translation bound, not an executable jump command or required vertical-height increment.

`FULLBODY_PREDICTION_VALID=false`; final impact evaluation is `NOT_RUN_TERMINAL_REJECTED`. PhysX validation, native adoption, native hit, Recovery success and complete5 are NOT_RUN. No qualified reference was exported and no 128-case evaluation was started.

## Reproduction and actual exits

From the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_push_off_window_v1.py tests/test_push_off_support_qp_v1.py tests/test_implicit_command_support_qp_v1.py tests/test_admitted_arm_support_qp_v1.py tests/test_action_conditioned_base_v1.py tests/test_contact_reach_bound_v1.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.rollout_push_off_prefix_v1 --contact-window --output /tmp/contact_window_new_attempt.json
```

The recorded output path was `reports/contact_gated_push_20261118/RESULTS.json`; the script refuses to overwrite an existing attempt. `FINAL_TESTS.txt` records **29 passed**, exit **0**. `RUN.log` / `RUN.exit` record the CPU rollout, exit **0**, 12.209 s. `COMPARISON.json` and `analyze.py` preserve the paired analysis. `MANIFEST.json` hashes the implementation and verifies protected inputs unchanged. The command error is the existing float64 position-command encode/decode check in radians, not a new native execution audit.

## Next step

Keep the default unchanged. Before another candidate, examine whether a whole-body angular-momentum objective can prevent the increasing negative pitch while generating the required COM motion through actual support forces. Then regenerate the arm endpoint against that action-conditioned base and independently repeat all trajectory and final contact checks. This is a hypothesis supported by the paired model response, not evidence of native benefit. More support duration alone is not the solution.
