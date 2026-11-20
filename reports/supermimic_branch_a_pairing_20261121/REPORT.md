# Fixed-case policy and world model audit

The historical R8/A8 candidate v16 completed 25 of 128 cases. The initialization completed 26. Two cases became complete and three lost completion. Two runs of the same v16 weights yielded identical returns in every case. The subsequent per-head refresh reused those weights and its existing proposal counters. Its two refresh names do not establish an additional R8 or A8.

| Same 128 cases | Initialization | v16 |
|---|---:|---:|
| Qualified airborne contact | 74 | 73 |
| Ball-side completion | 54 | 50 |
| Complete5 | 26 | 25 |
| Mean native return | −10.8740 | −10.7314 |

The archived policy hashes and measured causal recovery latches identify both actors. On identical Approach-phase observations (512 rows), the normalized request RMS change was 0.000909 and the mean pre-clamp Gaussian KL per action dimension was 0.000167. On Recovery-phase observations (896 rows), the corresponding values were 0.001805 and 0.000657. These measurements compare the two composed policies. A separate post-Recovery/pre-Approach frozen evaluation is needed to attribute either change to one head.

## Critic

Each row uses 128 observed native rewards paired with the immediately preceding and following states in the same episode. Both critics were evaluated on the same rollout. The input encoder uses the matching checkpoint. The values describe the fixed training suite.

| Native step | TD MAE before refresh | TD MAE after refresh |
|---:|---:|---:|
| 0 | 0.05113 | 0.04683 |
| 256 | 0.11936 | 0.12055 |
| 400 | 0.06979 | 0.07128 |

## World model

On 320 shared windows under a second simulator seed and the historical case identities, 64-step ball velocity MAE fell from 0.5413 to 0.4473 m/s. Ball position error at the 95th percentile rose from 1.2948 to 1.3942 m. The online checkpoint's training history includes these case identities. A second evaluation uses 32 newly generated launch vectors disjoint from the training and formal test suites, with the checkpoints and launch vectors committed to a local manifest before simulation. Its manifest SHA256 is `543a168af3f5d5c94e566764a8972dbff3eb6066beb63083c91bb0ff78e82d8f`.
