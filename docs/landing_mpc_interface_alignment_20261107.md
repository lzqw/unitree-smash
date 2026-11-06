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
