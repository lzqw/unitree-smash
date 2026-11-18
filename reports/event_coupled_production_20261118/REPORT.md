# Event-Coupled V4 CPU closure report

This report records the bounded real CPU search after the three integration
fixes. No 128-case evaluation, RL training, or SuperMimic process was run.

## Changes verified

1. `StrikeStateImpactBuilderV1` now builds the mass matrix, racket-point
   Jacobian, support Jacobian, support mode, and effective-mass diagnostic from
   the terminal state of the same 18D rollout. Analytic impact output is nested
   under `final_impact`; it cannot overwrite path or terminal errors. A
   contact-position rejection is reported as `FINITE_MASS_MODEL_COMPUTED`, not
   as native collision verification.
2. The event layer now retimes the actual command/reference prefix for a hit
   time change, recomputes the causal ball state through the existing predictor
   callback, and carries takeoff, push duration, COM acceleration, pitch-rate,
   string `(u,v)`, and hit-time parameters. The changed prefix is re-rolled
   from the original state and previous command.
3. `EventWBCCommandGeneratorV1` invokes the existing `PushOffSupportQPV1` per
   node and installs its accepted leg targets as an external per-node 12D
   correction. The arm remains the single prescribed 6D command source. QP
   rejection remains fail-closed.
4. Adapter and rollout evidence are allow-listed. Caller-supplied finite-mass
   strings or cached impact results cannot mark a candidate valid.

## CPU evidence

The historical lane was replayed from the same initial state and previous
18D command. The no-WBC event search used six real full-body rollouts and the
WBC-connected run used one additional real rollout.

| run | calls | best position error | normal error | velocity error | finite-mass status | rejection |
|---|---:|---:|---:|---:|---|---|
| event search (`RESULTS_v14.json`) | 6 | 51.449 mm | 0.12367 | 1.8589 m/s | model computed | string contact mismatch |
| WBC-connected (`RESULTS_wbc.json`) | 1 | 87.741 mm | 0.13677 | 1.6350 m/s | model computed | string contact mismatch |

The event search changed the actual prefix. A hit offset of `+0.03 s`
produced `contact_index=36` instead of 33 and a different full-body terminal
state. Its position error was 171.580 mm, so it was rejected. The best result
remained the 51.449 mm candidate. The WBC-connected candidate was also
rejected. The later bounded event sweep tested the same event-time and COM
variables. Positive COM requests that would have raised the body were rejected
by the original joint target limits at node 3; the other event-time candidates
were 185.531 mm and 236.478 mm away. The earlier fixed-prefix reference
remains 58.598 mm.

The common first physical rejection is:

```text
ball_string_contact_position_mismatch
```

The same-candidate evidence contains a finite mass model with inverse normal
effective mass `24.08320702160579 kg^-1`; this is a CPU model diagnostic. It is
not `FINITE_MASS_VERIFIED`, because no legal string contact occurred.

## Gate result

```text
POSITION_FEASIBLE       = false
NORMAL_FEASIBLE         = false for the best candidate (0.12367 > 0.12)
ATTACK_GEOMETRY_FEASIBLE= false
ANALYTIC_GOAL_VALID     = false
TASK_FEASIBLE_RACKET_VELOCITY = false
FULLBODY_PREDICTION_VALID = false
PHYSX_REFERENCE_VALID   = NOT_RUN
NATIVE_DEBUG_ELIGIBLE   = false
NATIVE_HIT              = NOT_RUN
RECOVERY_SUCCESS        = NOT_RUN
COMPLETE5               = NOT_RUN
```

The 3 mm contact gate was not met. Therefore there is no qualified candidate
and no safe basis for a native PhysX replay in this run. The failure is now
localized to the event-conditioned full-body reach/contact state, rather than
to missing finite-mass construction or a permissive evidence adapter.

## Tests

```text
40 passed
```

Covered evidence overwrite protection, same-candidate admission, support
Jacobian validation, event parameter changes, WBC callback shape, planner
gates, and the existing coupled rollout contracts.

## Remaining engineering action

The next search must change the physical takeoff state through a WBC task that
keeps the racket inside the authored string surface. The present event layer
is executable and causal, but the tested WBC correction moved the racket away
from the ball. It must be retuned using the measured 51.449 mm contact vector;
no tolerance or physical limit was relaxed here.
