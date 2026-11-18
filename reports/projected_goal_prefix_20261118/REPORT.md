# Explicit projected-goal full-prefix refinement

## Change and provenance

`refine_contact_event_v1 --projected-goal` explicitly applies the verified goal
projection to a complete predecessor. It first restores the old exact target
and command/reference history using the unchanged continuation gate. It then
checks projection SHA256, source-prefix SHA256, ancestry, identical causal ball
and incoming velocity, identical event time, finite unit normal and a fresh
forward ball-task check. Only goal normal/velocity and their analytic evidence
change. The initial physical state, previous actual command, causal ball,
contact point, dynamics, constraints and full-prefix commands are preserved.
The first fresh rollout must reproduce the predecessor q/v at every node.

The old and new goal residuals are NOT treated as physical improvement. A new
baseline is computed under the new goal; all reported command effects below
compare against this baseline. The original goal is retained in projection_audit.

## Actual CPU run

One deterministic 24-parameter full-prefix refinement, global rate basis,
33 control intervals, 160 integration substeps per interval. Original command
position and arm tracking constraints, terminal-task non-regression, measured
tracking linearization reserve and fixed backtracking [1, .5, .25] retained.
No random search or native evaluation. Runtime 226.5243 s, 27 fresh rollouts,
exit 0. Original prefix q/v reproduced exactly.

| Actual predicted residual, under SAME projected goal | Baseline | Selected half-step |
|---|---:|---:|
| Contact position (mm) | 119.511374 | 111.588458 |
| Normal vector distance | 0.225464827 | 0.187343268 |
| Racket velocity (m/s) | 2.779230081 | 2.775041653 |

This is an actual 18D target change and CPU-model response change, not merely
renaming a target. Maximum final-target change from baseline: 0.05 rad.
Maximum selected arm tracking error: 0.249940200650 rad, below unchanged 0.25.
Maximum execution mapping error: 4.44e-16 rad, below original 2e-6.

The full-step proposal failed at node 16 with tracking 0.250009161052 rad and
was rejected; its partial prefix was not accepted. The independently replayed
half-step passed all 33 path intervals. Logs preserve this rejection. The
empirical local tracking reserve was 0.00015045078445 rad; it is not a guarantee,
and the final physical bounds were not changed.

43 CPU regression tests passed, exit 0. Tests cover target/event/history
preservation, incorrect causal ball/event rejection, freshly invalid task
rejection, and the existing full-prefix constraints and reference gates.

## Status

The conditional analytic goal is valid. The complete CPU model path passes.
However, final position 111.6 mm exceeds 3 mm, normal 0.187 exceeds 0.12,
and velocity error 2.775 m/s exceeds 0.75 m/s. Therefore complete hitting
FULLBODY_PREDICTION_VALID remains false. No reference is qualified or installed.
PHYSX_REFERENCE_VALID, NATIVE_HIT, RECOVERY_SUCCESS and COMPLETE5: NOT_RUN.
No native/128-case evaluation, training or SuperMimic operation occurred.
Protected physics, Recovery/base controller, scorer, weights and source inputs
remain unchanged (manifest hashes verified).

## Next work

Goal adaptation is now connected to actual full-prefix refinement and yields
some geometric response improvement. The speed deficit barely changes: this
is evidence against treating target projection alone as the solution. The
selected path remains near the arm tracking bound; further work must improve
physical reference tracking/whole-body motion, with complete revalidation,
rather than relax that bound or repeat fixed-normal velocity sampling.
CPU model agreement with PhysX remains unproven. Do not infer native success
or actuator safety from these approximate model checks.

Stage teacher_debug; formal_eligible=false; formal_acceptance=false;
formal_cases_read=false. The overarching predictive-control goal is incomplete.
