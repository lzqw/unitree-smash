# Open-loop world-model diagnostic

The saved `checkpoint_000500.pt` was inspected offline on CPU. No Isaac episode was started and no weights were modified. For valid reset-free replay windows, the trained one-step model was rolled out autoregressively and compared with recorded next observations.

| Horizon | Mean state MSE | Last-step MSE | Interpretation |
|---:|---:|---:|---|
| 0.32 s / 32 steps | 0.176 | 0.638 | usable only as a short local model |
| 0.96 s / 96 steps | 14.367 | 67.045 | already outside the recorded-state distribution |
| 1.60 s / 160 steps | 54.816 | 159.998 | unusable for value or actor targets |
| 3.20 s / 320 steps | 216.228 | 621.175 | fully divergent open-loop rollout |

The model's teacher-forced one-step fit can remain small while the autoregressive state drifts. The 3.2 s run therefore exposed a model-horizon problem: the critic was trained on highly corrupted imagined states, so its loss stayed around 1,600--3,500 and the actor-quality gate correctly kept actor updates at zero.

This rules out the immediate fix of lowering the critic-loss threshold or forcing actor updates. The next experiment must either improve long-horizon world-model consistency (for example, a longer/multi-scale supervised window with explicit event/clock handling) or restrict imagination to a validated horizon and use a separate mechanism for delayed task credit. The 3.2 s run is not a policy-performance result.
