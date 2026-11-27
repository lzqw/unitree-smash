## Native Recovery tail value on R13 held cases

All three critics use the same frozen observation normalization and PPO01 behavior. The diagnostic compares predicted discounted native return with full episode Monte Carlo return, including the low-root terminal adjustment. The two-archive CPU fit combines same-policy R12 and R13 training lanes, excludes both held splits and keeps each episode in its own GAE lane.

| Fit archive | All active MAE | Pre-failure MAE | Pre-failure signed bias |
|---|---:|---:|---:|
| R12 | 9.405 | 7.761 | +6.893 |
| R13, current Actor training | 8.798 | 9.070 | +7.978 |
| Both, offline candidate | 8.383 | 8.394 | +7.745 |

The shared comparison has 6,397 active frames on 31 held lanes. The pre-failure subset covers the last 48 active ticks including the event on 24 cases with first body contact or low-root termination. The pooled fit improves absolute error relative to the current tail at both scales; it still overestimates pre-failure returns, and the R12-only fit has lower pre-failure error. Clean strict recovery has support from only two held cases. The pooled fit remained offline during this block.
