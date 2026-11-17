# 9df85ec follow-up: analytic task-valid rate retarget and strict reference-export gate

Baseline Git commit: `9df85ecfa91ad82bf4bafbf544da9dfa434ac853`.
Branch: `review/impact-feasible-rate-and-bundle-gates-20261118`.

**Scope: source-level experimental changes only.** Neither this review environment
nor the GitHub connector can run the user's installed IsaacLab interpreter or
launch the user's PhysX server. Full branch CPU tests, archived trace replay,
native 2-lane probe, 128 cases, PPO/BC, and SuperMimic work are **NOT_RUN**.

## Existing evidence (not new results)

The baseline reports 109 passing CPU tests, 4/24 pose-feasible contact events,
0/24 velocity-feasible events, 0 full-body combined prefixes, and 0 valid
offline bundles. Re-running the same historical action with the CPU solver
at 160 integration substeps still fails at node9, so matching PhysX's
62.5-us integration period alone does not validate the model. Planning p95
~15 s is offline telemetry, not a rejection criterion.

## Changes in this branch

1. `quadsmash/control/impact_feasible_rate_v1.py` adds an opt-in,
   deterministic bounded six-DOF joint-rate search. It computes the
   actual analytic outgoing shuttle velocity using the **existing**
   `ImpactOutcomeEvaluator`, rather than claiming success from
   matching the prior inverse-model racket speed. The only candidate
   eligible for further investigation must satisfy, simultaneously,
   structural string contact, legal return, original attack5, and
   target landing. All six motor speed bounds are unchanged (5 rad/s).
   The original inverse-model request and error remain stored.
   Neither analytically accepted rates nor desired outgoing speed
   are whole-body/PhysX execution certificates.
2. The new `impact_feasible_rate_retarget` config and explicit
   `--impact-feasible-rate` CPU replay option are **off by default** and
   require the existing anchored search. The desired outgoing target
   is transported through the same causal time-specific metadata.
   A successful analytic alternate rate is marked
   `task_feasible_velocity_retarget=True`, and still must satisfy
   Hermite bounds, all 18D physical checks, the actual terminal
   contact, and final analytic return.
3. `ContactFirstFullBodyEvaluatorV1` uses the unchanged 0.75 m/s
   **actual terminal tracking** criterion against the selected task
   rate only if that rate has independently passed the analytic task.
   The error against the original inverse racket request is saved in
   `original_requested_racket_velocity_error_m_s`. This is a
   **retargeted task**, not falsely claiming the prior velocity
   request became feasible.
4. `FullRacketRallyControllerV4` requires a final actual-terminal
   3-mm impact position and all original ball-side flags before
   admitting a newly selected anchored 18D plan. Missing/NaN
   evidence fails the single lane with frozen base fallback rather
   than changing the physical threshold or terminating other lanes.
5. `scripts/replay_contact_candidate_gate_v1.py` now only exports
   an offline reference after the controller actually **selects**
   the complete 18D plan, checks its unique exact physically replayed
   target sequence, and retains the FINAL impact checks. A
   full-body-evaluator callback alone is insufficient; such a candidate
   may still fail final impact. Raw public reconstruction remains
   `state_source=public_reconstruction` and cannot pass native handoff.
6. `scripts/offline_reference_bundle_v1.py` contains the strict
   final-export provenance helper. It requires both approved plan
   and full-body prefix and original attack/net/landing checks.

## New CPU tests (not yet executed by this review)

- `tests/test_impact_feasible_rate_v1.py`: feasible task retarget,
  infeasible bounded velocity, no outgoing impulse, invalid speed input.
- `tests/test_position_anchored_contact_search_v1.py`: synthetic
  end-to-end retarget through the selected-velocity/final-impact gate.
- `tests/test_contact_first_fullbody_evaluator_v1.py`: selected-vs-original
  tracking error, unchanged .75 m/s rejection, missing adapted target.
- `tests/test_offline_reference_bundle_v1.py`: reject early physical
  approval, invalid final ball return, invalid 3-mm gap, missing 18D.
  
On the original isolated teacher worktree, use:

```bash
git fetch origin review/impact-feasible-rate-and-bundle-gates-20261118
git switch review/impact-feasible-rate-and-bundle-gates-20261118
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q \
  tests/test_impact_feasible_rate_v1.py \
  tests/test_position_anchored_contact_search_v1.py \
  tests/test_contact_first_fullbody_evaluator_v1.py \
  tests/test_offline_reference_bundle_v1.py \
  tests/test_contact_first_trajectory_search_v1.py \
  tests/test_joint_prefix_execution_v1.py
```

After passing tests, use the **exact existing frozen two-lane inputs**
in the baseline REPORT; give each output a fresh path, no data overwrite.
Add `--impact-feasible-rate` to the baseline fixed-configuration
`--anchored --reachable-impact --velocity-aware --replanning`
command, preserving `--joint-stops --prediction-substeps 160`.
Do NOT change the original asset/limits/scoring/case identifiers.
Report original-request error and newly adapted analytic task separately.

## Required remaining work — no physical closure claim

- **P0 trusted physics:** the three same-state actual/frozen/combined
  action replays and the 160-substep solver still cannot reproduce a
  complete original trajectory. Determine the true native PhysX joint
  velocity-limit response and foot-contact/solver parameters from a
  controlled from-reset ground-truth replay; do not clip `-53.7`
  to `-30` or add unconstrained braking impulses.
- **P0 joint objective:** determine whether any rate under original
  `5rad/s` bounds satisfies the FULL analytic return from the saved
  contact states. If not, move legal target, original authored string
  contact point and hit time within the unchanged task domain; then
  consider low-dimensional takeoff state changes. Do not improve only
  the reported error by renaming an achieved rate.
- **P1 full offline teacher:** implement from-frozen-initial-condition
  PhysX verification of immutable 18D reference bundles; compare
  actual previous final 18D target and full state at the handoff,
  replay one complete 8-s episode, verify racket string hit and
  original recovery/scorer. Keep native modes and nominal teacher
  separate; do not silently add teacher policy updates.
- **P1 causal base:** long-horizon base-motion extrapolation remained
  unreliable in historical validation. Use native-calibrated
  action-conditioned rollout only after physical model validation.
- **Compute:** ~15-s offline planning is telemetry; no 100-ms or
  10-ms online claim. Native inline anchored search remains disabled.
- **Protocol:** current branch is `teacher_debug`,
  `formal_eligible=false`, `formal_cases_read=false`.
  No PhysX-valid trajectory, native adopted hit, or `complete5`
  can be claimed from this source-only patch.

If tests fail, preserve all logs and keep candidate/recovery/final scoring
at NOT_RUN; fix the actual failure rather than changing hard limits.
