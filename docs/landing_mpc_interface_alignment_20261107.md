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
10 passed
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
