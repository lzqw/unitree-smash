# Teacher normal/sliding-contact follow-up — source-only experimental patch

Base Git commit: `8853ccc81f038694f878ad33c9ccb52b585cc9f2`.
Updated source branch: `review/teacher-normal-slip-closure-20261117`.

**Scope:** production function changes and source-level tests. This review has
not executed the user's IsaacLab Python interpreter or the native 2-lane/128
panels. No SuperMimic file/process, original controller, WBC, Recovery MPC,
joint/racket asset, scoring function, or hard collision threshold was changed.

## Rechecked source; important frame correction

The caller `FullRacketRallyControllerV4._install_contact_first_plan`
already returns future-base-local tool FK in
`kinematics_at_time_fn`, while `plan_anchored_events` compares against
`R.T @ (current_planning_target - predicted_base_displacement)`. This
pair has a consistent **future-body** contract; the original concern
that the caller returns an unrelated world-frame point does **not**
apply to this commit. The *separate* inverse-impact continuous optimizer
uses a world-frame FK callback, appropriately. No frame conversion
modification was made.

## Experimental code changes

1. `constrained_contact_normal_v1.py` adds deterministic SLSQP refinement
   with analytic normal and position-inequality Jacobians. All candidate
   states are verified using the **same exact FK** as the existing arm
   anchor; the immutable `q_anchor` is preserved. This is attempted only
   if the original 10-micrometre-slack nullspace method fails.
2. `position_anchored_contact_search_v1.py` now may allow a **maximum
   0.5-mm position budget** (or the already tighter original position
   budget when larger), still `<=3 mm` and inside original joint limits.
   The result is adopted only on unchanged normal tolerance success or
   material normal-error improvement (>= max(0.02, 5% relative)).
   Otherwise the old strict result is returned without sacrificing a
   precise position anchor. The whole contact search is still CPU-only
   and blocked from native inline execution by the existing runtime gate.
3. `unilateral_implicit_prediction_v1.py` adds substep-only contact-mode
   and slip-direction warm starts **only when opt-in joint stops are on**.
   A geometrically separating foot still initializes released. Sliding
   direction updates use a quarter-step rather than a half-step to limit
   antipodal oscillation. The original contact, torque, normal-force,
   friction, position/velocity, stop complementarity and numeric
   acceptance tolerances are unchanged. An 80-iteration failure still
   raises; there is **no fallback certification**.

## New source tests / execution status

- `tests/test_constrained_contact_normal_v1.py`: coupled position/normal
  synthetic task, 0.5-mm cap, incompatible target rejection,
  malformed position budget rejection, and feasible-anchor retention.
- `tests/test_unilateral_implicit_prediction_v1.py`: warm-started Coulomb
  slip, negative force-velocity work, original dynamic/friction
  residuals and invalid hint rejection.
- A local synthetic SciPy mathematical smoke (not a run of the source
  branch's full pytest) confirmed a 0.30-rad orientation adjustment
  within a 0.300-mm position cost under the same nonlinear inequality,
  and a 1.57-rad target remained infeasible within 0.5 mm.

**The GitHub branch's CPU pytest is NOT_RUN in the review environment.**
The 89 passing CPU tests and p95=420.85 ms belong to **base `8853ccc`**.
Do not re-use those counts or runtime measurements for this source patch.

On the preconfigured original IsaacLab interpreter (read-only to
SuperMimic), run:

```bash
bash scripts/python.sh -m pytest -q \
  tests/test_constrained_contact_normal_v1.py \
  tests/test_position_anchored_contact_search_v1.py \
  tests/test_unilateral_implicit_prediction_v1.py \
  tests/test_contact_first_trajectory_search_v1.py \
  tests/test_contact_first_fullbody_evaluator_v1.py \
  tests/test_joint_prefix_execution_v1.py
```

Only after CPU tests pass, run the **same frozen input** as G0:
`python -m scripts.compare_position_anchor_history_v1`.
Compare the original `0.1681725686 mm` position anchor, accepted
contact-position error, and vector normal error `0.7137`. Distinguish
"best unadopted diagnostic normal" from actual fully feasible impact.
Do not expect an improvement or claim one before that replay finishes.

For the solver, replay **the same** recorded three action sources
(historical actual18, frozen base18, combined arm/legs18), with
`pair_rr_calf_replay_v1 --joint-stops`. Record the exact first failure,
active modes/directions and residuals around the historical node9.
Only a complete original-horizon pass, not nine passing nodes, counts as
dynamics-feasible. If it still fails, classify as model nonconvergence;
do not loosen tolerances.

## Required remaining gates to finish the actual closed loop

1. **G1 geometry/normal:** first preserve original `q_anchor`; show
   contact-position <=3 mm, original requested normal vector error <=0.12,
   request-vs-achieved racket speed with existing limits. A submillimetre
   location alone is **not** a strike.
2. **G2 dynamics:** original actual18 trace must replay without numerical
   failure and without original physical violations; compare force/contact
   states against real native traces. Then a new combined candidate must
   pass the full 18D trajectory, finite-mass forward impact and final
   3-mm ball-side rule. All fail-closed gates retained.
3. **G3 accuracy:** replace unvalidated ~0.8-s constant-acceleration
   base trajectory by a tested causal action-conditioned rollout. Do not
   inject historical future native states into control.
4. **G4 runtime:** profile inclusive planning time; the prior p95
   420.85 ms violates the desired <100 ms local-plan budget. The new
   constrained refinement may be more expensive; benchmark it. Native
   inline anchored planning remains disabled. A separate planning
   worker must first support immutable snapshot / reset generation /
   state fingerprint / actual prior18 / stale-result rejection /
   current-state revalidation / Recovery priority.
5. **G5 first native:** only after G1-G4 are satisfied, freeze exactly
   one candidate, start one isolated 2-lane 800-tick native development
   probe, and collect applied 18D command 2e-6 round-trip, racket
   position/normal/speed, PhysX string hit, outgoing shuttle flight,
   Recovery ownership and native complete5. Otherwise mark NOT_RUN.
   Only an adopted, valid native hit can justify a later 128-case
   development evaluation.

**No REAL closed-loop success is claimed by this commit.** Do not
start a native probe, 128-case panel, a parameter sweep, or any training
from this branch until those gates pass.
