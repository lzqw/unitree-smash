# SuperMimic probe and WM status, 2026-11-20

## Scope

This report covers the offline contact-window dataset, the high-epsilon native
probe collection, and the V13 native-response calibration. No Recovery or
Approach Actor/Critic update was started.

## Data and windows

- Frozen PhysX archive: 102,400 control-boundary transitions, 800 ticks by
  128 lanes.
- Event records: 1,026 episodes/events with train, development validation and
  final holdout identities kept separate.
- Core event window: 4 control ticks before the event and 8 after it.
- Full WM context: 4 ticks before and 64 after it.
- The archive has no PhysX substep truth. The new high-epsilon probe contains
  substep traces and is used only for native-response diagnostics.

## Native probe

The completed high-epsilon region contains 7 paired samples from one contact
specification. All 7 minus and plus branches report an analytic racket impact
at control tick 0. First impact substeps are recorded in the region artifact.
The plus-minus native ball response norms are:

```text
0.0054706, 0.0187744, 0.0142132, 0.1178561,
0.0230743, 0.0070241, 0.0195362 m/s
```

The region artifact is
`/home/autodrive/LZQW/experiments/supermimic_native_contact_regions_20261120_eps005_high/spec_00_tick271_contact.pt`.

## V13 calibration

The 256-update GPU fit used 24 calibration samples, 7 development samples and
8 final holdout samples.

| Split | Final response RMSE | Zero-response baseline | Minimum cosine | Maximum relative error |
|---|---:|---:|---:|---:|
| Development | 0.0270241 | 0.0271134 | 0.2590 | 0.99955 |
| Final holdout | 0.0081691 | 0.0081819 | -0.08345 | 1.00844 |

The candidate does not pass native direction or amplitude acceptance. Its
predicted V13 residual is nonzero, but the residual is almost the same for the
minus and plus branches. The body-model predicted robot endpoint does change,
with plus-minus robot-state norms from about 0.0034 to 0.0341, but this change
is not being converted into the measured ball-response difference.

An additional reverse split was fitted with the 7 high-response samples as the
calibration set and the 24 earlier samples as development validation. Its
validation RMSE was `0.041491`, versus a zero-response baseline of `0.041494`.
This rules out the simple explanation that the 256-step optimizer budget was
too short. The learned head still predicts a response close to zero on unseen
contact states.

On the high-response region, the plus-minus final joint-target difference is
about `0.004` in the perturbed joint and the Body WM endpoint state difference
is nonzero. However, the V13 local TOI feature difference is only about
`1e-4` on average and the V13 residual difference is about `4e-5`, while the
native ball response reaches `0.118 m/s`. This makes insufficient action
identifiability in the Body WM and probe timing the leading hypothesis. It is
not evidence that the analytic contact gate is missing all impacts.

## Diagnostic correction

`fit_supermimic_native_probe_response.py` now summarizes impact metadata across
every control tick in a rollout. It reports the first predicted impact, object
id, TOI, impulse and residual, and counts any active racket impact across the
whole rollout. This prevents a contact at an early tick from being reported as
an analytic miss merely because the final tick is non-contact.

The fixed diagnostic output is saved at
`/home/autodrive/LZQW/experiments/supermimic_native_response_fit_v13_20261120_highdev/diagnostics_fixed.json`.

## Action-to-racket Oracle A/B

The offline Oracle compares the same paired probe through PhysX endpoint
states and through `PhysicalWorldV2` endpoint predictions. On the 7 contact
lanes, Body WM gain ratios have the following medians:

| Quantity | Body-WM/native gain ratio | Direction cosine median |
|---|---:|---:|
| Racket position | 0.085 | 0.359 |
| Racket linear velocity | 0.142 | 0.908 |
| Racket angular velocity | 0.029 | 0.909 |
| Racket normal change | 0.011 | 0.860 |

The diagnostic artifact is
`/home/autodrive/LZQW/experiments/supermimic_action_to_racket_oracle_20261120.json`.
This confirms a large action-to-racket amplitude loss, especially in angular
velocity. The leading bottleneck is therefore the Body WM execution response
and probe timing, rather than V13 head capacity alone.

## Data-collection change

Event-aligned native probe specs now support `action_hold_ticks`. A signed
perturbation can be held for a causal 2--8 tick window while the nominal policy
continues to generate the base request. The default remains one tick for
backward compatibility. This enables the next probe set to measure action
delay and actuator response instead of only a single late perturbation.

## Verification status

- Contact event dataset tests: 5 passed.
- Diagnostic script syntax check: passed.
- GPU V13 fit: completed, candidate rejected by native direction and
  amplitude criteria.
- WM validation receipt: not issued.
- Contact validation receipt: not issued.
- Actor/Critic policy learning: not started.

The current next engineering target is to separate the Action to Racket
response from the Racket to Ball response. Increasing V13 updates without
that separation is not expected to improve the native response metrics.

## Action-identification collection profile

For the next native collection, the event core uses four control ticks before
the recorded event and eight ticks after it. The larger world-model context
remains four ticks before and 64 ticks after the event. The paired action
perturbation is introduced four ticks before the event and held for four
control ticks. This separates actuator delay and racket motion from the
single-tick contact response. The resulting probes are packed by
`build_supermimic_action_identification_dataset.py` into a dataset explicitly
marked as unavailable for policy-distribution Replay.

The collection is still an identification experiment. It cannot issue a
contact receipt or unlock Actor/Critic training until native response direction
and amplitude pass on an independent holdout.

The first held collection completed one spec at event tick 267 with four
contact lanes, a four-tick lead and a four-tick hold. It produced four
immutable paired records and was packed as
`supermimic-action-identification-v1`, with `admitted_to_policy_training=false`.
The final endpoint Body-WM/native gain ratios remained low, but the temporal
rollout showed the expected delay effect. Across the five recorded control
steps, median angular-velocity gain increased from 0.034 to 0.314 and median
linear-velocity gain reached 0.300 at the first step. This confirms that a
single endpoint metric was hiding actuator response over the held window, while
also showing that the current Body WM still attenuates the response.

The temporal Oracle artifact is
`/home/autodrive/LZQW/experiments/supermimic_action_to_racket_oracle_20261120_lead4_hold4_v2_temporal.json`.
It is diagnostic only and does not issue a native or contact receipt.
