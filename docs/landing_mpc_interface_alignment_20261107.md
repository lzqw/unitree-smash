# Landing MPC interface alignment — 2026-11-07

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
