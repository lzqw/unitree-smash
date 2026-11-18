# Reference-preserving joint prefix relinearization

This is offline teacher development, not a native evaluation or qualified teacher reference. Base commit 53cf985d83fa35de0002d160c7cd9fd84656aefb. No training, SuperMimic mutation, asset/physical-limit/scorer change, Recovery change or 128-case evaluation.

## Actual changes

The coupled optimizer can now continue from an explicitly supplied previous verified model prefix (`--previous`). It restores the complete planned arm reference, rather than resetting it to the original nominal reference or using measured joint positions. Every node exports its planned reference. Legacy migration is allowed only when original baseline arm commands match the frozen source within 2e-6 rad, and reconstructed tracking errors match recorded values within 1e-9 rad. Explicit and per-node histories must agree exactly. Inputs and derived evaluator state are hashed for response-cache isolation.

The opt-in `--backtracking` uses fixed scales 1, 1/2, 1/4 and stops at the first path-valid total-cost improvement. Each candidate starts from the original state and previous actual command and undergoes the complete 29-interval prediction. No partial failed prefix is executed or reused. Defaults remain unchanged. All 27 model calls in the final run were fresh; no cache was used.

## Evidence and results

The initial full-step attempt is preserved under `attempts/full_step01/` with source snapshots, raw result, log, hash and exit code. It failed at zero-based node 15: arm tracking 0.258014106643 rad exceeds the original 0.25 rad gate. The final run reproduced this failure, then independently verified the half-step.

| Quantity | Previous selected prefix | New half-step prefix |
|---|---:|---:|
| Contact position error | 425.248711 mm | 378.013569 mm |
| Normal vector error | 0.141881005 | 0.186708865 |
| Requested racket velocity error | 4.186762787 m/s | 4.188053281 m/s |
| Normalized total cost | 180.182172869 | 150.251071085 |

Position improved by 47.235142 mm, but normal and velocity tracking regressed. `improved=true` is only the local optimizer's total-cost comparison, **not** task success or all-objective improvement. The original final gates remain 3 mm, normal 0.12 and velocity error 0.75 m/s; all three fail. Analytic impact is not evaluated because terminal contact fails. No qualified reference is exported or installed.

29/29 model intervals passed path checks. Maximum command mapping round-trip error: 4.4408920985006262e-16 rad; this is CPU float64 mapping evidence, not native actuator execution evidence. Maximum arm tracking error: 0.247233587554 rad. Fullbody task acceptance is false; PhysX reference, native hit, Recovery and complete5 are NOT_RUN. CPU contact predictions remain experimental and are not claimed equivalent to PhysX.

## Validation and timing

84 tests passed, exit 0 (`TESTS_FINAL.txt`). Full CPU run: 204.876043 seconds, exit 0, 27 fresh full-prefix model calls, two proposal checks. No realtime claim. The initial two added synthetic tests failed because their unit-scale residual and regularization produced a smaller step than the fixture assumed; the fixtures now use explicitly scaled residuals. That failure output is preserved. Production thresholds were not changed to pass tests.

`COMMANDS.txt` contains exact reproducible commands. Source and protected hashes were checked after completion. Candidate input hashes and restored reference evidence are in BACKTRACK.json and MANIFEST.json. SuperMimic was observed alive during this CPU run and was not operated on.

## Next work (not executed in this run)

1. The weighted terminal cost accepts worse normal/velocity tracking. Add task-aware acceptance or constrained terminal refinement before claiming a joint improvement; keep this rejected-as-teacher candidate as diagnostic evidence.
2. Position error remains 378 mm after actual full-prefix propagation. Diagnose terminal reachability and base/arm timing on the same predicted trajectory before spending more fixed-normal rate trials.
3. Preserve explicit references in every subsequent relinearization and independently revalidate every complete sequence; no inferred or nominal history replacement.
4. Independent PhysX calibration remains required to establish model accuracy. CPU path validity is not a native certificate. No native candidate run until final contact and task gates pass.
