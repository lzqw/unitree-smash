## Effect of first-body-contact termination on learned Recovery actions

The absorbing-risk fit reused PPO01, the R13 native collection, trained ensemble, tail value, start schedule, 20-update budget and retention procedure. On native training-side observations, we replayed both conservative candidates and measured the difference from the actual normalized PPO01 action after clipping. The native baseline reproduced within 2e-6. The cosine below compares the two candidates' flattened action changes; it does not measure the agreement of a gradient with PhysX.

| Training subset | Independent cases | Action-change cosine | RMS difference between candidates |
|---|---:|---:|---:|
| Last 48 active ticks before body contact, primary | 61 | 0.9970 | 0.00075 |
| Last 48 active ticks before body contact, h8d | 66 | 0.9965 | 0.00075 |
| Clean strict recoveries, primary | 9 | 0.9979 | 0.00026 |
| Clean strict recoveries, h8d | 6 | 0.9984 | 0.00029 |

First-contact termination produces only a small change in the resulting conservative actions on this panel. Native64 is running. The next data collection is queued on the same R13 training cases with the conservative candidate to measure the native response to this update and obtain the executed target trajectories. The fixed128 and independent512 cases are excluded from this collection.
