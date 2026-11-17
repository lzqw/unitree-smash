# Velocity-aware offline teacher and prediction-contract audit

Baseline424e60c. **Code/test progress; offline full-body closure NOT_COMPLETED. No PhysX run,128 evaluation, RL/PPO/BC/distillation or SuperMimic action.** Existing independent teacher worktree/dependencies retained. Prior untracked assets/configuration, attempts and audit directories untouched.

## Implemented

- Added `velocity_aware_contact_pose_v1.py`: bounded joint pose+rate optimization at a pose-feasible seed. Exact FK must keep position within0.5mm, original normal-vector tolerance0.12, authored position bounds and supplied six rate bounds. No automatic3mm expansion. Best feasible evaluated iterate retained; solver status and original anchor remain separate.
- Wired it behind `velocity_aware_pose` / `--velocity-aware`. Normal/position gate precedes this solver. Actual achieved velocity, original requested velocity, optimized requested velocity, joint rate saturation and optimistic directional support bounds are separately logged. A support bound ignores acceleration/torque/trajectory and is not an execution certificate.
- Explicitly enabled existing `--reachable-impact` for one fixed CPU configuration alongside anchored/replanning/velocity-aware. It optimizes legal outgoing goal/contact UV in the existing authored strip and original event windows. Added explicit `accepted_semantics=analytic_ball_side_only`, `analytic_target_valid` and strict3mm `robot_contact_kinematic_valid`; analytical acceptance is not robot feasibility. Final actual impact/physical gates remain unchanged.
- Added explicit prediction settings to the native adapter and offline replay: substeps and joint-stop opt-in are forwarded to the actual evaluator. Constructor permits1..160 substeps; defaults remain4/stop-off. Every result/config records model version, integration dt and the fact that current velocity response is **postsolve rejection only**, not PhysX-equivalent braking.
- Added functional offline reference bundle export/read/handoff validation. Requires complete bounded18D targets, previous actual target, state25/24, contact time, controller parameters, hashes and predictor contract. Archive SHA256 checked, overwrite refused, reconstructed state cannot pass a native-state handoff, previous command checked at2e-6rad. CPU evidence cannot mark native stages successful. Candidate exporter is wired to accepted fullbody results in replay, but this run had **zero such results**, hence no full18D bundle was fabricated.

## CPU tests

`ALL_TESTS_DELIVERY.txt`:109 passed, exit0 (earlier full run also109; TEST_EXIT.json records that command). New tests exercise redundant-configuration speed improvement, immutable anchor/request, directional support bound,0.5mm cap, tampering/missing command history, and explicit160-substep diagnostic contract without claiming PhysX equivalence. Existing teacher arbitration/history/scoring export and physical regressions retained. These synthetic tests are engineering evidence only.

## Fixed-input replay, not a success-rate evaluation

Same saved2 lanes/8 fixed planning events/24 time candidates as previous round. Configuration: `--anchored --reachable-impact --velocity-aware --replanning --joint-stops --prediction-substeps 160`.

Position+normal feasible candidates: **4/24**, versus previous3/24. Speed-feasible candidates: **0/24**. New candidate full18D prediction calls: **0**. Original speed tolerance remains0.75m/s; no projected speed target renamed as success.

For corresponding historical pose candidates, requested-speed errors under the **newly optimized goals** are approximately:

| Lane, planning clock, relative contact | New-goal speed error |
|---|---:|
|1,2.17s,0.63s|1.6606m/s|
|0,2.49s,0.29s|1.9707m/s|
|1,2.49s,0.31s|1.7533m/s|

These are not fixed-target causal improvements: inverse goal, contact point and sometimes seed changed. Local velocity-pose optimization alone improved the first from1.6728 to1.6606m/s; other listed local solves retained their previous best. All four real local solves hit the fixed40-iteration budget; retained states are independently FK-checked, not called converged/global optima. A diagnostic transport defect was found after that replay: the searcher's metadata whitelist dropped the newly added original-target fields. It is now fixed and regression-tested. The original replay with null fields is preserved; VELOCITY_REQUEST_AUDIT.json reconstructs those errors from saved causal targets, body transforms and achieved velocities without changing any candidate. For the three historical counterparts, errors relative to the OLD request are2.9181/2.5062/2.3620m/s, versus1.6606/1.9707/1.7533m/s for the NEW goals. Lower new-goal error must not be advertised as better tracking of the old request.

Lane0 at2.49s/+0.29s has an optimistic current-configuration directional shortfall about0.9499m/s, already above0.75. This is useful local velocity evidence, not a proof that all alternative configurations/base motions are impossible. Other candidates can pass a directional upper bound while still failing the full3D velocity task.

The reachable inverse path executes many internal flight/impact residual evaluations. Planning mean11.90s/p95 15.08s across8 calls, with10,656 internal impact evaluations, rather than prior470ms. This is recorded separately in RESULTS.json, **not treated as an offline acceptance failure**. No algorithm was changed based on native scores.

## Actual physics audit and160-substep diagnostic

Verified frozen protocol `configs/g07_full_racket_eval_v5_contact_recovery_origin0.json`: physical dt62.5µs, control dt10ms,160 physical substeps. `full_racket_cfg` and the origin0 config chain use this final setting; the inherited CoupledEnv2ms default is not the evaluation setting.

Model4-substep dt2.5ms is40times coarser. A new explicit160-substep comparison used the **same three historical/base/combined commands and initial state**, without altering limits, force model or contact thresholds. All three still failed solver at node9 (10th control interval). Therefore temporal resolution alone does not establish correct prediction or reproduce FL calf's native−30rad/s response. The earlier4-substep result reached node10 and then predicted−53.7rad/s; these are different failures and both are retained.

Local IsaacLab actuator configuration and [official velocity-limit documentation](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim) describe physics-solver joint speed limiting/braking, with effort/solver interactions. They do not justify arbitrary unlimited braking reactions in this CPU surrogate. **No post-integration velocity clipping or uncalibrated speed reaction was added.** The requested PhysX-consistent speed response remains unresolved, explicitly labeled in the prediction contract.

Source audit: ground static/dynamic friction0.8; CPU contact coefficient0.65. Effective contact-pair coefficient, counterpart material and combination effects remain UNVERIFIED; neither value was adjusted. `PHYSICS_CONTRACT_AUDIT.json` records sources/hashes, not a claim of a newly loaded native scene.

## Offline validation scope and gates

The bundle is a candidate interchange/guard, **not a completed PhysX replay runner**. It demands a from-frozen-initial-state replay and matching native handoff; copying q/v is not enough. Automatic independent PhysX validation and original8-second scoring have not been exercised or certified. With no speed-feasible/full18D candidate and untrusted historical prediction, no native candidate was frozen or launched.

- POSITION_FEASIBLE: recorded per candidate.
- NORMAL_FEASIBLE:4 candidates together with position.
- VELOCITY_FEASIBLE:0.
- ANALYTIC_RETURN_VALID: analytic target evaluations only; no final actual robot-return candidate passes the whole chain.
- FULLBODY_PREDICTION_VALID: no new candidate; historical160-step diagnostic fails.
- PHYSX_REFERENCE_VALID / NATIVE_ADOPTED / NATIVE_HIT / RECOVERY_SUCCESS / COMPLETE5: **NOT_RUN**.

Recovery, physical assets, actuator limits and original scoring sources remain unchanged (PROTECTED_SHA256.json). No completion marker. The incomplete speed-response model and native replay integration are not presented as finished.

## Evidence

REPLAY/REPLAY_AUDITED preserve fixed-configuration CPU outputs (the latter adds original-request audit and bundle metadata, no algorithm retuning). PAIRED_160 preserves all failure traces. Prior4-substep baseline remains in reports/normal_slip_validation_20261117/PAIRED_DELIVERY.json.gz. One audited replay launch used a wrong interface filename and exited1 before planning; its log is retained, followed by corrected same-configuration run. Exact commands/exits in COMMANDS.md.
