# SuperMimic autoregressive world-model fit run

## Configuration

- Branch: `supermimic-structured-history-event-metrics-20261114`
- Code commit: `a7a5918`
- PPO initialization: `artifacts/ppo_full_racket_residual_v4/policy.pt`
- 512 native Isaac environments on `cuda:0`, 100 Hz, 8 s episodes
- World-model fit window: **0.96 s / 96 transitions**
- Imagination horizon: **3.2 s / 320 transitions**
- `--world-autoregressive-fit` enabled
- World batch 256, one world update per iteration, imagination batch 1024
- Actor/critic/world learning rates: `1e-5 / 1e-4 / 5e-5`
- Actor gate: full native timeout plus critic warmup and critic loss <= 50

## Training result

The process received SIGTERM at iteration 877 and saved `checkpoint.pt` and `training_status.json`.

| Metric | Value |
|---|---:|
| Native iterations | 877 |
| Native transitions | 449,024 |
| World updates | 782 |
| Critic updates | 1,560 |
| Actor updates | **68** |
| Last world loss | 5.526 |
| Last critic loss | **1.878** |
| Last actor gradient norm | 5.361 |
| Last policy anchor penalty | 0.0582 |
| Peak allocated GPU memory | about 5.17 GB |

The actor gate opened at iteration 810 after 483 full timeouts and 520 episode records. The first critic-only loss was about 0.805; subsequent critic losses stayed around 0.7--2.6, so the value-quality gate remained healthy. Actor updates did not show numerical explosion during the observed 68 updates.

The native event totals at the stop point were:

| Event | Count |
|---|---:|
| Episode records | 520 |
| Full 8 s timeouts | 483 |
| Qualified airborne string contacts | 76 |
| Ball-side complete except recovery | 13 |
| Recovery successes | 242 |
| complete5 | 6 |
| Frame hits | 165 |
| Body-ground records | 429 |
| Rally faults | 76 |

These event counts were collected before the post-update episode batch completed. They are training-stream counts, not a frozen 128-case evaluation and not evidence of a new success rate. The run proves that autoregressive 0.96 s world-model fitting makes the 3.2 s critic target numerically usable; it does not yet prove that the actor update improves the native task.

## Checkpoints and status

- `checkpoint_000500.pt`: periodic checkpoint before actor updates.
- `checkpoint.pt`: final checkpoint at iteration 877, actor updates 68.
- `training_status.json`: status `signal`, with the saved counters.

The next required measurement is a native post-update episode batch or a frozen 128-case evaluation using this saved actor. No claim of performance improvement is made from imagined return alone.
