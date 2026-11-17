# Experimental fullbody admission — fixed CPU panel

Base: `4d903a3`, continuing the task-constrained search introduced after `910d6f2`. Independent teacher worktree `/home/autodrive/LZQW/unitree_smash_teacher_eval`. No native simulation, policy training, SuperMimic changes, or 128-case evaluation in this round.

## Actual change

Added opt-in `experimental_spheres` fullbody prediction and `--experimental-spheres MANIFEST` to the offline replay entry. The adapter uses the previously audited authored foot spheres, original joint position/speed constraints, Head_lower sphere response and hip/mount response. Original default prediction paths are unchanged. Contact iterations are bounded at 2000, tolerance unchanged. Physical timestep is 62.5 microseconds, 160 subdivisions of the original 10ms interval.

The adapter checks asset/dependency hashes. It divides impulse momentum/cone/negative-normal residuals by the microstep duration before applying the original generalized-force and friction gates. Original mixed complementarity and fixed-point residuals retain their original solver thresholds and units. Missing/nonfinite/negative diagnostic values fail closed. Every microstep checks authored nonfoot geometry; an interior violation or nonfoot load cannot disappear at the interval boundary and pass. This is an approximate CPU contact model, NOT a PhysX equivalence claim or native execution certificate.

## Fixed replay results

Exit 0. Same input SHA256, original 8 planning calls / 24 contact events; no expanded fixed-face samples or parameter scan. Control sources stayed hash-identical throughout this run (`FREEZE_CHECK.json`). Search still finds the same two analytic joint-task states. Lane0 remains rejected by full Hermite joint-rate limits. Lane1 enters fullbody prediction.

| Gate | Result |
|---|---:|
| Original fixed event candidates | 24 |
| Joint pose + bounded endpoint rate + analytic ball task | 2 |
| Fullbody trajectories attempted | 1 |
| Lane1 integration/path constraints | **29/29 intervals completed** |
| Fullbody including terminal task admission | **0** |
| Qualified reference bundles | **0** |
| PhysX reference / native hit / Recovery / complete5 | **NOT_RUN** |

Previously that Lane1 candidate stopped at node2 due to contact/motor nonconvergence. Now the entire 0.29-second, 18D prefix runs with causal frozen-base leg commands, commanded arm trajectory, isolated histories and all original gates. Maximum original joint speed violation is 1.9984e-7 rad/s, position violation 1.7910e-12 rad; motor torque violation zero, nonfoot penetration zero and nonfoot loaded substeps zero. Generalized-force balance residual is 4.1606e-10. Maximum arm tracking error is 0.236965 rad, below the existing 0.25rad path gate.

**Terminal task fails:** position error 0.764064m, normal-vector error 0.403364, task-adapted velocity tracking error 4.884297m/s. Original ideal velocity error remains separately recorded as 4.176559m/s. Rejection is `actual_racket_terminal_residual`; no fullbody-valid or native-valid reference was exported. Completing integration is not passing the full task.

This exposes a planning/execution mismatch hidden behind the earlier solver failure. The planned endpoint had 0.48371mm kinematic position error, but that assumed future base transport and ideal arm tracking. In the complete CPU trajectory those assumptions do not hold together. A read-only FK decomposition (`TERMINAL_ATTRIBUTION.json`) separates an arm-angle tracking contribution of norm0.29109m and remaining base-transport contribution of norm0.89836m. These are signed vector contributions with cancellation, NOT additive percentages or native errors. The loaded-model FK agrees with the evaluator terminal callback to 1.52e-7m; a gross terminal coordinate conversion error is not supported by this check.

All29 actual predicted states and final18D commands are retained in `REPLAY.json.gz`, along with initial geometry, terminal state, selected endpoint and causal task metadata. They are a **rejected diagnostic trajectory**, not teacher demonstrations.

## Tests and timing

- Initial adapter/fullbody/bundle/ownership suite: 63 passed, exit0.
- Final suite adds clean-boundary/interior-collision regressions: **65 passed**, exit0.
- Search/anchoring/task constraints: **48 passed**, exit0.
- One attempted regression command named nonexistent test files: exit4, no tests ran. `SEARCH_TESTS.txt` is retained; corrected command/results are separate. Do not count the erroneous invocation as passing.
- Source diff check: exit0.
- Eight complete planning calls: 3.13–48.26 seconds (raw durations in SUMMARY). Offline generation only; no realtime claim. The longest includes fullbody verification. No online control loop was blocked by this offline process.
- SciPy reported a trial point outside SLSQP variable bounds and clipped its optimizer trial; raw warning preserved. This is not postprocessing/clipping simulated joint velocity.

## Next step, driven by this evidence

1. Replace the candidate's coarse future-base transport with action-conditioned fullbody transport, and re-solve terminal contact against that trajectory. The current 764mm terminal miss must not be concealed by precise IK against a different assumed base state.
2. Account for implicit PD tracking in the arm prefix, retaining original effort/rate/position limits; bounded endpoint rates alone do not ensure the endpoint is reached. Lane0 still independently fails its Hermite rate gate.
3. Revalidate a revised complete sequence from the current state/history before any export. Only if complete physics plus final3mm contact and forward ball task pass should a separate frozen native reference be considered.
4. Preserve Recovery, scoring and SuperMimic. No evidence in this round warrants changing them. Native closed-loop success remains NOT_COMPLETED.

Artifact whitespace check note: source-only `git diff --check` passed. The later all-artifact staged check returned2 for a preserved blank line in the failed pytest log and a context-space line inside the archived patch. These raw evidence files were not rewritten to conceal their original output; this is not a source/test failure.
