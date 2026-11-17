# Causal contact and scoring follow-up

Baseline: 44dc534. Fast-forward integrated user review branch through 29dac2a55468a9f78c8f0d0c2730b051b4668487. Workspace: /home/autodrive/LZQW/unitree_smash_teacher_eval. No tracked dirty changes were present before integration; pre-existing untracked assets/robots/piper_standard/configuration, attempts/ and audit/ were left untouched. No SuperMimic changes, training or native runs.

## Implemented

- Integrated removal of legacy intercept displacement only from the causal time-varying target branch. Legacy base/fallback retains its offset. Integrated strict first-rally/first-jump join and original independent/complete5 scoring.
- Added a two-stage contact gate: only a geometric position mismatch within the existing 35 mm kinematic envelope may proceed to mandatory joint physical replay. Gross mismatch, nonapproaching contact, out-of-court prediction and other failures remain rejected. Missing/nonfinite gap never qualifies for deferral. The final impact evaluator still uses 3 mm; it is not widened.
- Joint candidates with impact evaluation must supply physical terminal position, normal and velocity, even if coarse impact passed. A boolean accepted result cannot bypass the final-state collision check. Both coarse and final outcomes remain in metadata. Normal/position/velocity/force/torque/ground hard constraints are unchanged.
- Impact results now retain contact error and actual contact tolerance, including on rejection.
- Probe saves raw rally and jump records before joining/scoring, preserving failure evidence. Export no longer fabricates `ground_init=True`; missing/false native ground initialization is rejected.
- Added reproducible offline replay CLI using the production controller, causal frozen-base leg reference and full-body gate. It reads observations only through the first historical planning event, never recorded future commands. Public state reconstruction is explicitly labelled, not native truth. This diagnostic adapter does not supply finite-mass refinement and must not certify accepted trajectories for execution.

## Tests

User-requested five-file regression after integration: 40 passed, exit 0 (BASELINE_TESTS.txt).
Final seven-file regression including full-body and unilateral physics: 65 passed, exit 0 (CPU_TESTS.txt, CPU_EXIT_CODE.txt).
New tests cover coarse defer -> strict final acceptance/rejection, missing physical terminal state, gross mismatch bypass prevention, and native ground-init preservation. These are CPU contracts, not native success evidence.

## Offline replay (one current configuration; no scan)

Saved native source: reports/joint_loop_round_20261117/attempts/probe01/native. Original candidate weights/config, 202 observed ticks through first planning event at 2.01 s. Input hashes in OFFLINE_REPLAY.json. Command exit 0.

Lane 0 still considers 54 candidates: 46 kinematic task residual, 4 joint-step limit, 1 unreachable velocity, 3 predicted out-of-court. For the three candidates reaching impact evaluation, contact error is 0.000376257 m (0.376 mm), down from the old 100.23 mm duplicate offset. All three pass geometric contact but predict upward outgoing z velocity 3.920–4.077 m/s and large lateral y velocity 5.946–5.973 m/s. Predicted landing y is 6.035–6.126 m, outside the court. These are analytic predictions, not native measurements.

Lane 1 considers 54 candidates: 51 task residual, 3 unreachable velocity. Best position error 0.0365454 m still exceeds the unchanged 0.035 m threshold. No relaxation or extra samples were introduced.

Both lanes rejected all candidates. Physical evaluator calls=0, accepted prefixes=0. Thus this replay demonstrates removal of duplicate target compensation, but does not validate the online 18D execution/PhysX contact loop, finite-mass response or Recovery handoff. The newly deferred 3–35 mm route is covered by CPU tests; this recorded sample does not exercise it. No saved historical jump record was fabricated to repair the old probe.

## Stop boundary / next work

Native small probe: NOT_RUN, because offline candidates remain rejected. Full128: NOT_RUN. New native success rate: NOT_RUN. No claim of improvement over 29/128, 20/29 or 10/128.

First next task is task-aware reachable normal/velocity generation: these candidates geometrically hit but send the ball upward and sideways. Retain original requested normal/velocity and propose bounded physically achievable task directions; validate actual output with the current collision model and full-body dynamics. Do not increase the candidate count to hide this issue. Lane 1 requires reachability/earlier preparation diagnosis, not a larger tolerance. Low-dimensional takeoff/WBC optimization remains subsequent work, not implemented or validated here. Recovery and Head_lower clearance limits remain unchanged.

Flags: evaluation_stage=teacher_debug; formal_eligible=false; formal_acceptance=false; formal_cases_read=false.
