# Inverse-impact targets and final-command history

Baseline 53299e8. No native evaluation, no 128 panel, no training, no SuperMimic modification or launch. Existing untracked assets configuration, attempts/ and audit/ were preserved.

## Implemented control changes

`inverse_impact_target_v1.py` generates one deterministic task target per candidate time, with existing drag, gravity and court geometry: landing at x=target upper bound minus 0.1 m, y=0, ball-radius height; desired outgoing vz=-1.01 m/s. A bounded least-squares solve (60 evaluations maximum; vx 0.1–30, vy -20–20 m/s, flight time 0.05–2 s) solves for the remaining velocity and flight time. These are target-generator bounds, not changed actuator limits. Normal is along desired outgoing minus incoming velocity. Racket tangential velocity matches incoming tangential velocity, giving zero tangential slip in the infinite-mass seed; the existing collision-plus-drag evaluator independently checks the result, including legal return, attack5 and target landing. An unsuccessful solve or forward check is explicitly rejected. This conservative seed may miss other achievable impacts; it is not an exhaustive optimizer or finite-mass feasibility certificate.

Production integration is opt-in via `ContactFirstSearchConfig.inverse_impact_targets=True`, requiring impact evaluation. The legacy branch remains unchanged by default. The task-directed branch does not choose the closest-to-current normal or project the requested normal onto the achieved one. It uses bounded task-space joint velocity least squares, all six arm joints, and one velocity scale (1.0), rather than increasing the number of amplitude candidates. Base linear/angular point velocity compensation is retained. Requested normal error and requested velocity error remain distinct from projected residuals. Each candidate is still subject to the original position, speed, contact and full-body checks; no final contact tolerance is relaxed.

A contact-time-specific inverse-target failure is counted/rejected without aborting all other candidate times. An invalid nominal seed can still reject the lane; no exhaustive time/reachability claim is made. The low-dimensional jump event optimizer is NOT implemented this round.

`CausalFrozenLegReferenceFactory` now requires the physical evaluator to commit the combined final18 target after propagation. It encodes that command into next-step last_action, instead of recording the frozen base proposal. Missing and duplicate commits fail. Fresh candidate histories remain independent. No command is committed when propagation throws. `predicted_observation_frame_v1.py` reconstructs all 69 current-frame fields from Pinocchio Racket pose/twist and causal ball state, including relative tool angular velocity, presence and validity. Goal timing is retained from the measured observation, not hardcoded. Phase reconstruction preserves observed airborne-hit state; predicted ground contacts still use the existing contact model, not fabricated forces.

Impact rejection metadata distinguishes structural contact failures from approximate analytic outcome failures. The original exact physics/geometry and Recovery implementation are unchanged.

## Real CPU evidence

Final nine-file regression: **68 passed in 1.68 s**, exit 0 (`CPU_FINAL.txt`, `CPU_FINAL_EXIT.txt`). Tests include final18 history/commit ordering/candidate isolation, inverse target forward consistency, impossible downward return rejection, and retaining other contact times after one target failure. The initial test run had **1 failure / 40 passes**: the old recorded +0.79 s contact state fails the unchanged net check. This was retained in TEST_INITIAL.txt. The positive test now uses the already-existing +0.77 s candidate from the same causal saved observation; it does not claim all contact times are feasible.

Saved-native frame comparison at ticks 200/201/202 for both lanes (`FRAME_REPLAY.json`): maximum tool-position difference 7.58e-7 m, normal 1.51e-6, relative linear velocity 4.79e-7 m/s, relative angular velocity 9.37e-7 rad/s. Contact/presence/validity match. Root and ball are reconstructed from the saved public inputs; these comparisons are NOT an independent native-truth calibration. Stage/command-history tests are separate CPU checks, not covered by this 69D comparison.

## Frozen two-lane offline replay

`REPLAY_FINAL.json`, command exit 0. Same recorded first planning event and original base checkpoint; only the explicit inverse-impact flag enabled. No future recorded actions are supplied to the reference generator. Intermediate replay is retained in REPLAY.json: it exposed the one-time failure abort bug; final replay follows that engineering correction, not parameter scanning.

Lane 0 at +0.77 s: desired/forward outgoing approximately **[5.9044, -0.0585, -1.01] m/s**; predicted net clearance **0.07494 m**, landing **[3.899916, 0.00000083, 0.025] m**. This is an analytically legal target, not a measured hit. At +0.79 s predicted net clearance is **-0.03172 m**, so the target is rejected.

Final search: lane 0 reports 12 kinematic-task-residual rejections plus 6 rejected time/branch combinations for unavailable analytic targets. Lane 1 reports 18 kinematic-task-residual rejections. Counts are reduced by the single task-speed scale, not enlarged. **Physical evaluator calls=0; accepted prefixes=0.** Eight impact-model calls include target-generator checks; they are not eight dynamically feasible candidates.

All robot candidates remain rejected. Native contact, actual command adoption, 2e-6 native command replay, and Recovery handoff: **NOT_RUN**. No success-rate improvement is claimed. No native evaluation was started, as explicitly requested.

## Next required work

Task-direction generation and action-history consistency are implemented, but legal task targets are still kinematically unreachable under the current predicted base/configuration. Next analyze requested-vs-achieved normals, position, joint-speed saturation and causal preparation timing. Only after that evidence should low-dimensional support/push-off action optimization be added. Retain Head_lower clearance and final 3 mm contact gates. Do not broaden sampling or relax tolerances merely to obtain acceptance.

Flags: evaluation_stage=teacher_debug; formal_eligible=false; formal_acceptance=false; formal_cases_read=false.
