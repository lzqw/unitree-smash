# Hybrid TOI V13 engineering closure report

Date: 2026-11-19
Branch: `feat/supermimic-hybrid-toi-v13-20261119`
Code commit: `c0722fc`

## Code and tests

- V13 semantic version is explicit and checkpoint migration is strict. V1 to V2 head reinitialization is recorded and V2 shape mismatches fail loudly.
- Hybrid manifests are written at the checkpoint top level and checked by the runtime loader.
- V13 pure mode skips the legacy six-dimensional residual network. The analytic prior and V13 residual share the deferred TOI post-impact integration path.
- Native probe diagnostics retain impact information for every rollout tick, including object, impact time, sampled gap, TOI gap, impulse and residual response.
- Verified no-contact samples are accepted only when native contact metadata confirms no contact in both paired branches.
- Repository tests: `470 passed, 1 warning`.

## GPU model smoke

- GPU: `cuda:0`
- Physical and ball model loading: passed.
- V13 forward smoke: passed with schema `native-temporal-causal-racket-mode-impulse-residual-v13`, semantic version `2`, impulse residual input width `12`.
- Three-step GPU response smoke: finite output, elapsed `0.710 s` for the recorded batch.
- No Actor, Critic, or policy update was run.

## Native PhysX event-aligned probe

Output:
`/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261119_seed882073/gpu_probe_current_c0722fc_run225943`

- Spec: `spec_00_tick267`, Approach, lanes `96, 99, 102, 115`.
- Branches: `nominal_a`, `nominal_b`, `minus`, `plus`.
- All four branches completed with exit code `0`.
- Each branch captured `268` control ticks and `161` physics substeps for the paired prefix.
- Nominal repeatability was recorded by the collector. No actor, WM or critic update occurred.
- The probe is a four-lane diagnostic and does not satisfy the original-128-lane fitting contract or the minimum independent native receipt support by itself.

## Current V13 hybrid rollout comparison

The current migrated V13 candidate was evaluated from the same recorded start and command sequences for three control ticks. The candidate is:

`/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261119_seed882073/temporal_response_fit_v13_current_20261119_224046/action_ball_candidate.pt`

Candidate SHA256:
`5f252f3cc4c188b31f19b39f57b54f061545a642a695d3a62173d2303428f7d5`

Native paired ball-velocity response norms for lanes 96, 99, 102 and 115 were:
`0.16843, 0.23733, 0.04763, 0.01787 m/s`.

Predicted norms were:
`0.00311, 0.00125, 0.00193, 0.00344 m/s`.

The ball-velocity response RMSE was `0.08539 m/s`. Per-lane direction cosines were approximately `-0.079, -0.026, -0.638, 0.215`.

The analytic prior detected a racket impact on the third control tick for all four lanes, with continuous TOI gaps near zero. The predicted nominal trajectories were finite, but the plus/minus action response remained much smaller than the native PhysX response. This is a Body-WM/action-to-racket response mismatch and is not an accepted contact-direction result.

## Admission status

`FIT_COMPLETED_NOT_ADMITTED`

`NATIVE_PROBE_COMPLETED_DIAGNOSTIC_ONLY`

`CONTACT_BPTT_BLOCKED`

`LARGE_ACTOR_TRAINING_NOT_STARTED`

The next required data are additional independent event-aligned probe episodes and lanes, including verified no-contact pairs and multiple response magnitudes. The existing four-lane probe is retained as evidence and is not used to claim a passing receipt.
