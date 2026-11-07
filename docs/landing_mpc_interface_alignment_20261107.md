# Landing MPC interface alignment — 2026-11-07

> **2026-11-07 correction:** the measured-position-centered target envelope
> discussed below was introduced by our MPC, not by the original native
> actuator interface. Native arm software slew is relative to the previous
> final target (10 rad/s for this checkpoint), while physical arm speed stays
> limited to 5 rad/s. See [the command-history correction](landing_command_history_correction_20261107.md).
> Earlier references to this envelope as the native/hardware admission rule
> should not be used as that claim. Historical experiment results are unchanged.


## Why this change was needed

The native full-racket environment consumes each MPC action as a normalized
absolute joint-position target.  The support QP predicts a target from the
requested torque, while executor replay already clips that target to the
declared one-step joint-velocity bound.  Before this change, the evaluator
sent the unclipped QP target to Isaac.  The predicted trajectory and the
executed command therefore differed at the first applied step.

This is an interface-consistency issue, not a new performance result.  The
existing fixed-panel baseline remains **27/128 complete5**; the historical
28/128 touchdown-height result is still treated as a one-off auxiliary result.

## Change

`LandingSupportMPCV1.admit_command_target()` now defines the single-step
position target admission rule:

\[
q^{cmd}=q+\operatorname{clip}(q^{qp}-q,
 -\dot q_{max}\Delta t,\n+ +\dot q_{max}\Delta t).
\]

The same rule is used by executor replay and by the native evaluator before
converting the target to the original normalized absolute-position action.
The landing record now stores both the requested and admitted targets and
their maximum per-step changes.  This makes it possible to distinguish a QP
request that is not executable in one step from an actuator-tracking failure.

No actuator limits, physical asset, contact model, or task acceptance rule
were changed.

## Validation

The targeted CPU checks pass:

```text
12 passed
```

Python compilation also passes for the modified MPC and native evaluation
entry points.

## Native check after the patch

Two native checks were run with the same frozen policy and physical protocol.
The representative run used TRAIN 1, 67 and 95; the full run used the 26
previously identified ball-side-good/recovery-failed cases.

| run | complete5 | MPC records | applied first steps | target-error mean / median / p95 |
|---|---:|---:|---:|---:|
| representative 3 cases | 27/128 | 446 | 331 | 0.628 / 0.568 / 1.378 rad |
| full 26 cases | **27/128** | 4069 | 3040 | **0.504 / 0.410 / 1.117 rad** |

The previous representative telemetry was 0.831 / 0.841 / 1.404 rad.  The
interface fix reduced command error, but it did not rescue any of the 26
targeted cases: airborne contacts remained 74, conditional recovery remained
34, and complete5 remained 27.  This separates an execution-interface bug
from the remaining task bottleneck; the next work must identify the native
PD/contact response and recoverable-state coverage rather than merely
increase QP gains.

The evaluator now also records the predicted next joint position and compares
it with the next native state.  The first diagnostic run had only 38 usable
prediction comparisons because held-action records were not yet included in
that statistic; those samples had 0.847 rad mean and 1.412 rad p95 error.  The
held-action path is now instrumented as well, so the next run will provide a
dense prediction-error distribution before any response-calibration change is
accepted.

## Dense prediction diagnostic

The follow-up representative run used the same frozen policy, physical
protocol, selected cases (TRAIN 1, 67 and 95), and native MPC settings.  It
completed the full 128-case panel with no numerical failure and again obtained
**27/128 complete5**, 74 airborne contacts and 34 conditional recoveries.
There were 446 MPC records and 331 applied first-step actions.

With held-action records included, the measured next-state errors were:

| quantity | samples | mean | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| admitted-command to measured joint position | 360 | 0.628 rad | 0.568 rad | 1.378 rad | 1.519 rad |
| predicted next joint position to measured joint position | 360 | **0.506 rad** | **0.398 rad** | **1.343 rad** | **1.501 rad** |
| predicted error on applied/held steps | 328 | 0.467 rad | — | — | — |

The admission rule is active: the nine rows containing a target delta had a
requested maximum step of 0.842 rad on average, while the admitted maximum was
0.300 rad, matching the configured one-step velocity limit.  The dense run
therefore confirms two separate facts: the evaluator now sends the same
one-step-admitted target that replay predicts, but the native state still
deviates substantially from the predicted next joint state.  Since the task
result remains 27/128, reducing the command-interface mismatch alone is not
enough to rescue the targeted recovery failures.

This is diagnostic evidence, not a claim that all of the remaining error is
caused by implicit-PD response.  The current replay still approximates the
full-body/contact response: it scales joint acceleration after a PD torque
estimate and does not re-solve the floating-base acceleration and contact
forces.  The next structural change should therefore calibrate or replace the
native response model and keep terminal acceptance on the same predicted
trajectory, rather than continue scanning QP weights.

## Admission-constrained QP follow-up

The next structural change moved the one-tick native position-admission bound
into the QP itself.  For a measured joint state (q,v), the QP now intersects
its effort/joint-limit torque set with the torque set induced by

\[
q-\dot q_{max}\Delta t \le q^{cmd}
\le q+\dot q_{max}\Delta t .
\]

The evaluator still applies the same admission helper, but it no longer needs
to clip most QP requests after the solve.  The change is covered by the CPU
tests (`11 passed`) and keeps the full and reduced support-QP paths consistent.

The full 26-case native diagnostic was rerun on GPU0 with the same frozen
policy and physical protocol.  It again produced **27/128 complete5**, 74
airborne contacts, 34 conditional recoveries, and zero numerical failures.
There were 5516 MPC records, 2869 applied first steps, 2791 terminal
acceptances and 1445 solver failures.  The telemetry changed materially:

| quantity | before QP admission constraint | after QP admission constraint |
|---|---:|---:|
| command error mean / median / p95 | 0.504 / 0.410 / 1.117 rad | **0.451 / 0.395 / 0.919 rad** |
| predicted next-position error mean / median / p95 | — | **0.298 / 0.197 / 0.955 rad** |
| requested vs admitted maximum target step | often clipped after solve | **0.289 vs 0.289 rad** |

Thus the QP now requests commands that are executable through the declared
position interface, and measured prediction error falls substantially, but no
targeted recovery case is rescued.  This is a useful negative result: the
remaining bottleneck is not primarily post-solve target clipping.  It is more
likely the contact/impact approximation and the fact that the support QP's
predicted acceleration does not yet guarantee a sustainable native support
response.  The next MPC change should therefore focus on contact-mode
transition, impact velocity/impulse, and measured support feedback rather than
another gain sweep.

## Optional touchdown-impulse diagnostic

The controller now has an opt-in touchdown event projection.  When the
prediction adds a new foot contact, it solves a unilateral normal impulse for
that foot and projects the predicted downward normal velocity toward zero.
Negative impulses are clipped, and the native simulator remains authoritative;
this is deliberately not presented as a full impact or friction model.  The
default is disabled, so the established baseline is unchanged.

A representative GPU0 run with this projection enabled (TRAIN 1, 67 and 95,
same frozen policy and 128-case panel) produced **27/128 complete5**, 74
airborne contacts, 34 conditional recoveries and zero numerical failures.  It
applied four predicted touchdown projections.  Their downward normal speeds
were approximately 0.0006--0.166 m/s and were reduced to numerical zero, but
no targeted recovery case was rescued.  Command error was 0.463 rad mean and
prediction error was 0.247 rad mean in this run.

This rules out a narrow explanation in which only the predicted touchdown
normal velocity is missing.  The impulse model remains useful for keeping
future predictions physically better posed, but the next performance change
must act over the following hundreds of milliseconds: re-estimate the real
support set and support forces from native contact telemetry, update the
continuous COM/attitude reference, and keep solving until a measurable stable
window is reached.

## Optional support-force allocation trial

To address the next part of that hypothesis, the reduced support QP now has an
optional `force_distribution_weight`.  When enabled, it softly biases the
normal reaction variables of the currently measured contacts toward an
equal-share load of (mg/n_{contact}).  It does not create contacts, override
friction, or replace the hard floating-base dynamics; the default remains zero.

The representative GPU0 trial used weight `0.05` with the same TRAIN 1, 67 and
95 cases and the same frozen policy.  It completed with **27/128 complete5**,
74 airborne contacts, 34 conditional recoveries and zero numerical failures.
There was no task-level gain.  This means that simply preferring equal normal
loads is insufficient; the missing information is the future support/foot
motion and the actual native response, not only the instantaneous force split.

The next controller revision should therefore use measured support force as a
feedback signal over time: detect unloading, update the viable contact set,
move a swing foot when the current support margin is no longer recoverable,
and evaluate the resulting trajectory after the native position/PD interface.

## Receding-feedback trial without fixed first-step hold

The fixed-hold path was then removed for the same representative cases.  With
`hold_steps=0`, the controller read the native state and attempted a fresh
eight-step solve on every active control tick.  The result remained **27/128
complete5**, with 74 airborne contacts, 34 conditional recoveries and zero
numerical failures.  The run produced 516 MPC records, 138 applied first
steps, 292 solver failures and no terminal acceptances.

This separates two issues that had been conflated: fixed-command holding is
not the reason the task does not improve, but simply re-solving the current
fixed-contact QP is also not a viable recovery teacher.  Its feasibility drops
when the support mode changes, and the accepted first-step actions do not
create a sustained stable state.  The next implementation should use a
contact-mode state machine with explicit support-loss handling and a
mode-consistent terminal test, rather than call the same QP more frequently.

## Next evaluation

The next representative native run should compare the same landing cases with
the previous model-consistent MPC configuration and report:

- requested versus admitted target deltas;
- previous-target to measured-joint error;
- first-step application and terminal acceptance;
- non-foot contact timing and stable-window outcome;
- complete5 and recovery counts against the 27/128 baseline.

If the command error falls while the task result remains unchanged, the next
focus is the contact-mode and support-response model.  If it remains large,
the native implicit-PD response and torque/position interface need further
calibration before changing MPC weights.

## Optional hard non-foot-clearance constraint

The next diagnostic added an opt-in one-step hard lower acceleration bound for
near-ground non-foot links.  It uses the same clearance target as the soft
task, but requires the predicted vertical acceleration to be large enough to
reach the clearance target in one control step.  This was tested only as a
diagnostic; the default remains disabled.

The GPU0 representative run used TRAIN 1, 67 and 95, horizon 8, native
position admission, executor replay, touchdown recovery, continuous support
reference, a 5 cm clearance target, and the hard bound enabled.  It completed
the 128-case panel with **27/128 complete5**, 74 airborne contacts, 34
conditional recoveries and zero numerical failures.  The MPC produced 729
records, but **all 729 solves failed**, with zero terminal acceptances and
zero applied first steps.

This is a useful negative result: the one-step hard clearance requirement is
incompatible with the current support-QP authority and contact model at the
tested states.  Increasing its weight or making the same bound harder would
only reject more candidates; it is not a usable recovery controller in this
form.  The option remains available for future diagnosis, but it should not
be enabled in the main evaluator.

The next structural change should therefore be a contact-mode/support-state
transition mechanism: keep unloaded or airborne feet out of the support
constraints, add a new foot only when touchdown is predicted and physically
consistent, and evaluate terminal stability on the same replay trajectory.
The controller also needs a multi-step recovery reference or mode sequence;
a one-step non-foot clearance inequality cannot create the missing support
motion by itself.

## Bounded contact-mode fallback trial

After the hard-clearance rejection, the existing bounded contact-mode fallback
was enabled without the hard constraint.  When the current support QP became
infeasible, the controller tried one or two imminent touchdown feet as an
alternative support set, then continued through the same replay and terminal
checks.  This is still an event approximation; it does not inject a physical
impact impulse or force a foot to carry load in Isaac.

The GPU0 representative run used TRAIN 1, 67 and 95 with the frozen policy,
the native position admission rule, executor replay, support-margin gate,
continuous touchdown reference, and a 40-step first-action hold.  The fixed
128-case panel remained **27/128 complete5**, with 74 airborne contacts, 34
conditional recoveries, and zero numerical failures.  It produced 658 MPC
records, 403 applied first steps, 139 solver failures and 7 predicted contact
transitions.

The fallback therefore makes the solver enter more contact-mode alternatives,
but it does not yet create a new successful recovery.  The important negative
result is that a bounded mode switch alone is insufficient: the newly added
contact must also be realized by the native foot trajectory and support force
response.  The next implementation should expose per-foot load transfer and
post-switch state error, then keep or reject a mode using those measured
signals rather than accepting a contact flag as support by itself.

## Capture-point foot-placement trial

The next representative run enabled the existing horizontal capture-point
reference for inactive feet (`capture_foot_gain=1.0`, maximum offset 0.25 m),
with contact-mode fallback and the same native execution/replay settings.  The
goal was to test whether a subset of the failures could be recovered by moving
the next foot toward the measured COM velocity direction before it became a
support foot.

The fixed 128-case panel again remained **27/128 complete5**, with 74 airborne
contacts, 34 conditional recoveries and zero numerical failures.  The run
produced 642 MPC records, 371 applied first steps, 139 solver failures and 4
predicted contact transitions.  No new complete recovery was obtained.

This result makes a pure horizontal capture-point explanation unlikely for the
current selected failures.  The controller can propose a shifted swing-foot
target, but the native trajectory still does not establish a sustained,
load-bearing support state.  Future work should therefore prioritize measured
per-foot load transfer and a multi-step support-mode reference rather than
another capture-gain sweep.

## Native foot-force telemetry check

The native Pinocchio adapter previously populated `measured_force_world` only
when a positive contact-force gate was requested.  That made the default
zero-threshold MPC mode unable to inspect the raw four-foot load transfer,
even though the simulator exposed it.  The adapter now always records the
4x3 world-frame foot-force tensor when available; the threshold still only
controls whether the force gates the support mask.

The unchanged contact-mode-fallback representative run was repeated after
this interface fix.  It remained **27/128 complete5**, with 74 airborne
contacts and 34 conditional recoveries.  The run produced 658 MPC records,
403 applied first steps, 139 solver failures, and 7 predicted contact
transitions.  The new telemetry shows that predicted feet often have near-zero
load at the prediction instant and only sometimes acquire load on later
ticks; in other cases the measured support set changes again before the
held action expires.  This confirms that a predicted contact flag is not
equivalent to a realized load-bearing support mode.

The next controller change should use this telemetry to validate a mode
transition and to release/replan when the expected foot does not acquire load,
rather than treating the predicted mode as persistent for the whole horizon.

## Centroidal wrench allocation trial

The support QP now has an optional centroidal wrench allocation task.  It uses
the active feet's moment arms about the measured COM and softly targets the net
force required by the requested COM acceleration with zero residual roll/pitch
moment.  This is a force-distribution preference only; hard dynamics,
friction, actuator limits, and the native position interface remain unchanged.

With `force_wrench_weight=0.05`, the same representative GPU0 protocol
remained **27/128 complete5**, with 74 airborne contacts, 34 conditional
recoveries and zero numerical failures.  It produced 997 MPC records, 694
applied first steps, 142 solver failures and 5 predicted contact transitions.
No complete recovery was added.

The result rules out a purely instantaneous force-allocation fix for the
selected failures.  The QP can prefer a better contact wrench, but the native
position/PD execution still changes the support set over time.  The next
controller revision should use the measured load history to choose a
multi-step support mode and update the COM/foot reference after each realized
contact transition.

## Load-validated held-action trial

The evaluator now has an optional contact-mode lifecycle check for held MPC
actions.  In the representative trial, every expected support foot had to
carry at least 5 N upward force for three consecutive control ticks; otherwise
the held first action was released and the controller re-solved from the live
state.  The default remains disabled so prior comparisons are unchanged.

With the check enabled, the fixed panel remained **27/128 complete5**, with 74
airborne contacts, 34 conditional recoveries, zero numerical failures, 597
MPC records, 284 held/applied first steps, 174 solver failures, and 8
predicted contact transitions.  There were 33 per-tick mode mismatches and 10
early releases of a held action.

This confirms that the mode-validation signal is causally observable and that
the old held command can outlive its expected support mode.  Releasing and
re-solving alone still produces no task-level gain, so the next controller
must change the multi-step support reference and load-transfer objective after
the release; it should not simply lower the hold duration or force threshold.

## Load-aware continuous support reference trial

The continuous support QP now has an optional load-aware reference.  When
native foot forces are available, the support-anchor and measured-foot center
used by the critically damped COM reference are weighted by each active foot's
upward load (with a small force floor).  The contact mask itself is unchanged;
this only changes the desired COM/support reference.

The GPU0 representative run enabled this mode with a 2 N force floor and the
same contact-mode fallback protocol.  It remained **27/128 complete5**, with
74 airborne contacts, 34 conditional recoveries and zero numerical failures.
It produced 615 MPC records, 346 applied first steps, 136 solver failures and
5 predicted contact transitions.  No complete recovery was added.

This rules out a support-center weighting change as a standalone fix.  The
remaining controller gap is a multi-step decision problem: after a contact
transition, it must decide whether to keep, unload, or move a foot and then
recompute a feasible support trajectory from the resulting native state.

## Stronger landing-leg posture trial

To test whether the delayed thigh contacts were primarily caused by a weak
posture target, the landing-leg posture blend toward the verified standing
posture was increased from the usual `0.15` to `0.5`.  All contact-mode,
executor-replay, and support-margin settings were kept unchanged.

The representative GPU0 run remained **27/128 complete5**, with 74 airborne
contacts, 34 conditional recoveries and zero numerical failures.  It produced
658 MPC records, 362 applied first steps, 141 solver failures and 4 predicted
contact transitions.  No complete recovery was added.

This makes a weak static leg-posture blend an unlikely standalone cause.  The
controller needs to change the support mode and load transfer over time; a
single stronger standing posture target does not create the missing recovery
trajectory.

## 400 ms receding-horizon trial

The controller was then run with `horizon_steps=40` (0.4 s at the 100 Hz
control period), while keeping the verified fixed-policy protocol, executor
replay, support-margin gate, predicted touchdown fallback, and continuous
reference enabled.  This was intended to test whether the previous 8-step
(80 ms) horizon was simply too short to see the landing transition.

The complete fixed TRAIN128 run remained **27/128 complete5**, with 74
airborne contacts, 34 conditional recoveries, 0 numerical failures, 658 MPC
records, 403 applied first steps, 139 solver failures, 392 terminal
acceptances, and 7 predicted contact transitions.  The longer horizon did not
add a complete recovery or preserve a new one beyond the frozen baseline.

This is evidence against a horizon-length-only explanation.  The present
controller still rolls a fixed-contact support QP forward and only adds a
contact mode when the current QP is rejected; it does not jointly optimize a
contact-mode sequence, foot relocation, and load transfer over the 400 ms
window.  The next structural change should therefore be a bounded
mode-sequence/replan mechanism, validated by measured foot load and actual
state after each transition, rather than more scalar horizon or gain scans.

## Bounded contact-mode sequence search trial

The controller now has an opt-in bounded mode search. At each horizon stage,
when an inactive foot is within the existing touchdown height/velocity window,
the controller evaluates the current support mode and up to three imminent
contact-mode candidates even if the current fixed-contact QP is feasible. It
selects a candidate only when the one-step predicted risk score improves; the
next live tick still validates measured load and can release/replan. This is
the first change that searches a support-mode sequence rather than only using
a new contact after QP infeasibility.

The representative GPU0 run used the same frozen policy and 40-step horizon as
the preceding trial, with `contact_mode_sequence_search=true` and
`max_candidates=3`. The full fixed TRAIN128 result remained **27/128
complete5**, with 74 airborne contacts, 34 conditional recoveries, 0 numerical
failures, 703 MPC records, 447 applied first steps, 140 solver failures, 435
terminal acceptances, and 4 predicted contact transitions.

The mode search therefore did not yet produce a task-level gain. It did,
however, change the internal candidate evaluation and increase the number of
accepted first-step applications without changing the frozen student's ball
events. The next useful extension is not a larger candidate count: it is to
score candidates with a short load-transfer/stability objective and to replan
from the measured post-transition state, so that a predicted contact is useful
only when it actually acquires load and reduces the subsequent non-foot
contact risk.

## Hold-release on measured command tracking error

The native diagnostics showed that the held MPC target was not being followed
closely: representative cases had mean one-tick joint target errors of roughly
0.28--0.48 rad.  The evaluator therefore gained an opt-in release rule: while
an accepted first action is held, a measured maximum joint tracking error above
the configured threshold immediately releases the hold and forces a new solve
from the live state.  This tests the position-PD execution interface directly,
rather than assuming that a held target is realized.

With a 0.1 rad release threshold, the fixed TRAIN128 run remained **27/128
complete5**, with 74 airborne contacts, 34 conditional recoveries, 0 numerical
failures, 513 MPC records, 139 applied first steps, 294 solver failures, 7
terminal acceptances, and 89 predicted contact transitions.  The stricter
release reduced the number of held/applied plans and did not rescue a new
landing.

This is useful negative evidence: stale-target tracking is a real interface
error, but simply re-solving more often does not fix the task.  A successful
controller must change the predicted support/load-transfer trajectory, not only
refresh the same position target after tracking error grows.

## Full landing-posture reference trial

The upstream reactive-landing structure uses a loaded landing posture and a
settling reference rather than the strike controller's instantaneous joint
target.  As a direct position-interface test, the support MPC was run with a
full (`1.0`) blend toward the admitted standing leg posture after touchdown;
all other settings matched the 40-step fixed-policy trial.

The fixed TRAIN128 result remained **27/128 complete5**, with 74 airborne
contacts, 34 conditional recoveries, 0 numerical failures, 717 MPC records,
430 applied first steps, 168 solver failures, 419 terminal acceptances, and 4
predicted contact transitions.  A stronger static leg target alone therefore
does not supply the missing sustained recovery trajectory.

The next implementation should use the upstream phase idea more faithfully:
after measured touchdown, generate a time-varying COM/attitude and foot-support
reference, then update it from measured load and state.  Raising a posture
blend without a load-transfer/foot-relocation plan is not enough.

## Latched post-touchdown lifecycle trial

The evaluator now supports a separate post-touchdown lifecycle latch. After
`phase774` is first observed, the MPC can remain eligible for a bounded number
of real 100 Hz control ticks even if the public phase signal briefly drops.
Every tick still reconstructs the native state, updates measured contacts and
loads, and can release a held action; the latch does not replay one command or
invent a contact.

The representative run used a 300-tick (3 s) latch, the 40-step horizon and
the same selected TRAIN cases and support settings as the earlier fixed-policy
trial. The full TRAIN128 result remained **27/128 complete5**, with 74 airborne
contacts, 34 conditional recoveries, 0 numerical failures, 1111 MPC records,
597 applied first steps, 228 solver failures, 582 terminal acceptances, and 9
predicted contact transitions.

The lifecycle latch therefore did not add a task-level recovery in this
configuration. It does rule out a simple phase-dropout explanation for the
selected cases: extending the eligibility window increased actual MPC
coverage, but the resulting position/PD support actions still did not move
the fixed policy into a durable support state. The remaining priority is a
load-transfer and foot-relocation objective with measured post-transition
replanning, rather than a longer eligibility window.

## 2026-11-07: contact terminology and continuous-reference memory correction

The native flag previously called `geometric_contacts` is actually a force-norm
hysteresis observation (1 N on / 0.5 N off). Comparing it with upward force
above 5 N compares two force-based criteria, not geometry against load capacity.
The evaluator now also records `observed_force_contacts` with an explicit
semantic label; the old key remains only for backwards compatibility.
The completed 5 N / 1 N / 3-tick trial is still **27/128 complete5**, 74 airborne
contacts and 34 conditional recoveries. The 37 prior disagreements do not by
themselves establish a cause of falling.

The completed rate-scale 0.2 trial (`landing_mpc_ratescale02_20261107_01`) also
remained **27/128**, 74 airborne contacts, 34 conditional recoveries, NF0.
It evaluated all 128 episodes but enabled MPC only on TRAIN1/67/95. This option
clips requested targets in prediction/execution; it does not shrink the QP's
internal torque set by the same factor. Thus this is a post-solve admission
experiment, not proof of an internally optimized reduced-authority controller.
Compact, source-hashed counts are in
`landing_mpc_memory_followup_evidence_20261107.json`.

A new, source-confirmed issue was found before further tuning:

1. The evaluator constructs one `LandingSupportMPCV3`. Its `ContinuousSupportQP`
   held one COM reference position/velocity, contact history and anchor array.
   These were shared across the selected environments.
2. Each QP call advances that reference by 10 ms. A horizon with up to 40 stages
   advanced live memory by prediction time, and each alternative contact mode
   could advance it again. Rejected plans did not roll that memory back.
3. Resetting any environment reset the shared MPC reference for all cases.
4. Held commands skipped the live observation path, including the internal
   contact-state counter and continuous reference update.

This is a controller-memory bug, not evidence that any particular failed state
is recoverable. It also makes prior continuous-reference trials unsuitable as
clean tests of the intended per-environment closed-loop reference.

The correction stores live memory separately per environment. Each eligible
real control tick (including held-command ticks) advances it once. Planning
uses a separate branch and restores live memory even after an exception.
Alternative contact modes start from the same stage memory; only the selected
branch advances the next prediction stage. Episode reset clears only that
lane. Contact observations now update before the held-action early return.
Physical parameters, policy, success conditions and the QP cost are unchanged.
This does not turn the sequential QP into a jointly optimized hybrid MPC, nor
does it fix implicit-PD/contact model mismatch by itself.

Validation: 16 focused CPU tests passed (four new memory regressions, plus the
existing MPC/contact-state suite). Tests cover interleaved versus independent
lanes, selective reset, rollback after failed prediction, once-per-tick held
observation, and branch-independent reference timing. A production QP check on
recorded TRAIN1 observation tick371 confirms identical first targets for
requested horizons 8/40 and no change of live memory after planning; both
predictions rejected at stage1. That is a consistency check, not native success.
See `landing_reference_memory_cpu_20261107.json`.

The provisional COM-braking-offset experiment was stopped at tick285 with no
complete 128-episode result. Its partial files and source snapshot remain local
under `reports/landing_mpc_combrake02_20261107_01/`; the speculative offset was
removed. It was only a heuristic reference offset, not an implementation of
upstream VHSIP, and must not be reported as a completed VHSIP/MPC result.

Next native comparison: retain the original 27/128 frozen policy, use the same
TRAIN1/67/95 controller selection, 40-step horizon, original rate-scale 1.0 and
40-step hold as `landing_mpc_horizon40_20261107_01`, changing only reference
memory/observation semantics. Report per-case recovery, ball-side intersections,
solver exceptions and old-success retention. No improvement is claimed until
this full native run completes.

### Disagreement events and native rerun status

The original 37 differing force-contact records cover **17 TRAIN cases**.
A provisional clock comparison placed 26 before the first-episode non-foot
collision, five after it and six against cases without a recorded collision.
**Correction:** legacy MPC logs lack episode IDs and can include unscored later
episodes; these 26/5/6 counts are NOT verified same-episode timing attribution.
Only three records apply a feasible MPC first stage (TRAIN12 at 7.47/7.88 s,
TRAIN50 at 3.58 s). The remaining records are rejected at stage0 or have no
admitted support. This is retrospective timing evidence, not proof that the
threshold caused or prevented falling; held intervals were not fully logged
with both predicates in that old run. Full compact events are in
`landing_contact_disagreements_20261107.json`.

The corrected native comparison started on GPU0 on 2026-11-07 at approximately
10:09 Beijing time, code commit `95854e0`, output
`reports/landing_mpc_reference_isolated3_20261107_01`.
It uses the existing single-GPU budget guard, a 3600-second job ceiling and
102400 control-environment-step cap. It is a frozen-policy controller
comparison, not PPO training. At this report revision it is **running**, with
no final performance result; 27/128 remains the retained baseline.


### Clearance coordinates: confirmed model-state error, native test pending

The reduced QP calls `reference()` after taking a measured-state snapshot.
That reference computes posture height with a zero-root configuration and writes
Pinocchio's shared Data. The subsequent thigh-clearance task previously read
frame positions/Jacobians from that scratch data as if they were the current
state. It also combined a LOCAL_WORLD_ALIGNED point Jacobian with an untranslated
WORLD Jacobian derivative.

On recorded TRAIN1 tick371 the model's current FL thigh origin is approximately
0.13285 m high, whereas the legacy cache supplied -0.01025 m. This is a comparison
inside the same model on a recorded public observation, not a newly measured
native clearance or collision-surface distance.

The correction explicitly recomputes and copies point kinematics at the current
predicted q/v, including the moving-origin term in the Jacobian derivative.
It changes no physics, actuator limits, cost coefficients or success conditions.
`landing_clearance_coordinates_cpu_20261107.json` contains a same-state CPU
comparison on 61 recorded post-touchdown states from TRAIN1/67/95. Median absolute
legacy height error is 0.16261 m, maximum 0.52492 m. Local QP acceptance is 51/61
before and 50/61 after: correcting the equations is not evidence of better
native recovery or even a higher local solve rate.

Validation: 18 focused tests pass, including independent finite-difference
checks of point velocity/Jacobian derivatives and a regression through the
production QP after reference FK has modified shared data. Returned snapshots
are verified not to alias later FK operations.

The evaluator also now records episode IDs/control ticks and stops MPC work on
each lane after its scored first episode ends. Previously unscored subsequent
episodes could add solver records and, before per-lane memory isolation, affect
shared controller memory. Native first-episode performance remains the score;
legacy all-call totals should not be interpreted as first-episode totals.

These new clearance/logging changes are **not included** in the currently
running memory-only experiment (source `95854e0`). Finish that comparison first,
then test corrected clearance with the same policy and controller settings.
No new success-rate claim is made from these CPU corrections.


### One-step prediction error must be time-aligned

The held-action diagnostic previously compared every held tick to the original
one-step predicted joint position, even 40 ticks later. Those aggregate values
are not valid one-step model-error distributions. The next evaluator records a
prediction only at its intended next control tick; later held ticks report null
instead of reusing stale predictions. Torque-overlay trials are excluded from
this prediction metric until their distinct executed input is represented.
Command-to-current-joint target tracking remains a separate quantity. This is a
logging correction only, not a control change or evidence that the native model
is accurate. Historical dense prediction statistics must be remeasured.


### Guarded continuation queued

A finite continuation waits for the memory-only run to finish successfully, then
runs `landing_mpc_clearance_fixed3_20261107_01` with the same frozen policy,
TRAIN1/67/95 controller selection and physical/control settings. It verifies
source hashes before launch and stops on parent failure, numerical failure or
source change. The child uses GPU0's exclusive budget guard, 3600-second limit
and 102400 environment-step cap. It has **not yet produced a result**.

Local orchestration receipts are under
`reports/landing_correctness_followup_20261107/`; compact summaries will include
complete5, retained/lost/new cases, selected-case recovery and episode-ID scope.
This comparison tests clearance correctness plus diagnostic alignment; it still
uses a 40-tick held command and does not establish continuously replanned or
jointly optimized MPC recovery.

### Prescribed arm target contradicts the final command admission

A further CPU audit on the same 61 recorded states compares three local QPs
with identical loaded dynamics and hardware bounds. Each sample resets the
reference, so this is not a trajectory or native performance comparison:

| arm treatment | independently accepted local QPs |
|---|---:|
| prescribe the supplied arm goal before rate admission | 34/61 |
| prescribe the arm goal after the existing 10 ms admission | 50/61 |
| let the QP optimize the arm within original limits | 51/61 |

In 27/61 states the supplied arm target differs from what final admission would
send. Prescribing the unadmitted arm torque overrides the interface torque
interval, although the final target is still clipped later. Sixteen legacy QPs
were numerically solved but failed independent residual checks. Pre-admitting
the arm goal gains 16 accepted states and loses none, without increasing motor
limits or freeing the arm. This isolates an interface contradiction; it does
not establish stable landing. Eleven samples still fail with admitted arm goals.

Evidence: `landing_arm_admission_audit_20261107.json`, reproducible with
`scripts/audit_prescribed_arm_admission_v1.py`. This uses recorded public-state
reconstruction and reset-per-sample references, not native-state trajectory
replay. The two already running/queued native comparisons retain their declared
sources; arm admission is a separate subsequent change, not silently included
in those comparisons.

### Recomputed time-aligned joint prediction diagnostics

Using the first chronological clock segment per case in the legacy logs:

| run | old held/stale-inclusive mean (rad) | matched one-tick n | matched mean / p95 (rad) |
|---|---:|---:|---:|
| horizon40 | 0.16823 | 11 | 0.00504 / 0.00896 |
| command-rate scale 0.2 | 0.18306 | 18 | 0.00429 / 0.00855 |

Each value is the maximum joint-position error for one observation. The matched
set requires the record to occur exactly one control tick after an applied new
QP command; later held observations are excluded. These small, selected sets
only assess accepted-command one-step joint motion. They do not validate base
motion, reaction forces, impact prediction, rejected states or long horizons.
Legacy episode identities are inferred only from the first monotone clock
segment. Fresh explicit episode/tick telemetry remains necessary.

This evidence weakens the interpretation that all large reported errors were
one-step actuator mismatch. It instead motivates correcting arm command
admission and replacing long single-command holds with sustained feedback after
the in-progress controlled comparisons. Reproduction and hashes are in
`scripts/audit_landing_prediction_timing_v1.py` and
`landing_prediction_timing_checked_20261107.json`.

`AdmittedArmSupportQP` is now implemented as a separate candidate, preserving
all original limits and the original prescribed-arm behavior after admission.
It records both requested and admitted references, rejects invalid targets
rather than hiding them with clipping, and leaves the caller's target unchanged.
Two new regression checks bring the focused suite to 20 passing tests. This
candidate is not yet wired into the running/queued comparisons, preserving their
source hashes and isolating the effects. It must receive its own native trial
before being promoted as a useful recovery teacher.

### Memory-only native comparison completed

`landing_mpc_reference_isolated3_20261107_01` completed 102400 environment steps:
**27/128 complete5, 74 airborne contacts, 34 conditional recoveries, zero native
numerical failures**. All original 27 complete cases were retained; none gained.
Controller selection remained TRAIN1/67/95, not all 26 recovery failures.

| case | previous horizon40 stable window | isolated-reference stable window | final recovery |
|---|---:|---:|---|
| TRAIN1 | 0.206 s | 0.754 s | failed; root-height termination |
| TRAIN67 | 0.105 s | 0.176 s | failed; non-foot contact |
| TRAIN95 | 0.060 s | 0.065 s | failed; non-foot contact |

TRAIN1 improved a partial stability measure, but never completed the required
0.8-second window and still collapsed. Do not count delayed failure as rescue.
There are 648 controller records / 347 applied records / 159 failure records;
these legacy totals include later unscored episodes and clock-reset exceptions,
so they are not first-episode feasibility rates. Compact evidence and hashes:
`landing_reference_isolation_result_20261107.json`.

The next clearance-only comparison started at approximately 10:30 Beijing time
on GPU0 (native PID4083994), keeping the source snapshot taken before arm changes.
It includes explicit episode/tick telemetry and stops MPC in completed lanes.
No result is available yet.

After that process had initialized its controller, the admitted-arm QP was wired
into the next V3 implementation. Both live reference updates and speculative
planning now receive the same admitted arm goal. The focused suite has **21
passing tests**, including live-versus-planned reference agreement. This new
implementation has not yet been run natively; it does not change the already
loaded clearance-only process.

### Sustained feedback candidate queued after clearance comparison

`landing_mpc_admitted_arm_feedback3_20261107_01` is queued behind successful
completion of the clearance-only run. It uses the admitted-arm V3 code, removes
40-tick single-target holds (`hold_steps=0`, no hold-first-action), and keeps the
recovery lifecycle active after observed touchdown for the remainder of the
8-second episode (`latched_post_touchdown_ticks=800`). Actual support constraints
still use current contacts; latching the recovery lifecycle does not latch
physical support forces. The 40-stage prediction and original hardware/physics
are retained. It still uses sequential support QPs, not joint trajectory
optimization, and applying an admissible first stage does not certify the whole
recovery.

This is an integrated sustained-feedback candidate (arm admission plus command
feedback/lifecycle), not a one-factor ablation. It runs only after source hashes
and the prior completion receipt pass, under the same exclusive GPU0 guard and
3600-second/102400-step ceilings. Local receipts:
`reports/landing_admitted_arm_feedback_followup_20261107/`.
There is no native score for this candidate yet.
