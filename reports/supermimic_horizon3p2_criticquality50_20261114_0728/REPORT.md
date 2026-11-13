# SuperMimic 3.2 s imagination-window run

## Run

- Branch: `supermimic-structured-history-event-metrics-20261114`
- Code commit at launch: `fc1c439e529bbe33702ba09a667794a988de0ece`
- Initialization checkpoint: `artifacts/ppo_full_racket_residual_v4/policy.pt`
- Native environment: 512 Isaac/PhysX environments, `cuda:0`, 100 Hz, 8 s episodes
- World-model window: 0.32 s (32 transitions)
- Imagination window: **3.2 s (320 control steps)**
- World updates: 3 per iteration; imagination batch 1024; actor microbatch 64
- Actor/critic/world learning rates: `1e-5 / 1e-4 / 5e-5`
- Actor gate: at least one full timeout, at least 200 critic updates, critic loss <= 50

## Outcome

The run was stopped after iteration 831 with a clean saved history. It was not a formal 128-environment evaluation and no actor update was accepted.

| Native training-stream metric | Value |
|---|---:|
| Native iterations | 831 |
| Native transitions | 425,472 |
| Completed episode records | 520 |
| Full 8 s timeouts | 483 |
| Qualified airborne string contacts | 76 |
| Ball-side complete except recovery | 13 |
| Recovery successes | 242 |
| complete5 records | 6 |
| Frame hits | 165 |
| Body-ground records | 429 |
| Rally faults | 76 |
| Actor updates | **0** |
| Critic updates | 640 |
| World updates | 2,400 |
| Peak allocated GPU memory | about 5.17 GB |

The 76/520 and 6/520 counts are training-stream event counts, not a frozen-policy success rate. They show that the PPO-initialized policy produced nonzero native task events before learning, but they cannot replace the fixed 128-case evaluation.

## Diagnostic conclusion

The longer horizon fixed the previous temporal coverage problem: the first imagination update occurred only after full 8 s episodes were observed, and the 3.2 s rollout was long enough to include the task event horizon. However, the critic-only gate remained closed:

- critic loss at the first full-episode update: about `1963`;
- later critic loss range: roughly `1600–3500`;
- actor updates: `0` throughout;
- imagined return remained strongly negative.

Therefore this run demonstrates **temporal coverage**, not effective policy learning. Lowering the actor gate or counting critic updates alone would risk changing the initialized actor before the value target is reliable. The next debugging target is the critic/world-model target distribution and its drift during critic-only updates, with target statistics and model-step consistency added before another actor experiment.

The earlier 0.96 s run and this 3.2 s run should be kept as diagnostics; neither is a new formal performance result.
