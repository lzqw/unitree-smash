# First Recovery block on 64 deterministic native cases

Both candidates completed the full evaluation. The row ordering and lane-specific baselines are bound in the accompanying JSON. Counts for body contact and recovery are conditional on the 54 qualified contacts in each slot.

| Metric | Baseline in slot 1 | Conservative | Baseline in slot 2 | Moderate |
|---|---:|---:|---:|---:|
| Body-ground contact | 31 | 30 | 32 | 28 |
| Strict recovery | 3 | 4 | 3 | 6 |
| Relaxed recovery | 9 | 10 | 8 | 10 |
| Ball-side complete | 46 | 45 | 46 | 45 |
| Robot net fault | 0 | 0 | 0 | 0 |

PPO01 remains the incumbent. Both candidates failed the no-loss ball-side gate on A-991101-002. First contact position and the times of first contact and release match across all four records. Both candidates crossed the net and landed at x = 4.0350 m. The two baseline landings were x = 3.9809 and 3.9198 m.

The native rule latches the last touching velocity on the release substep. For this case first contact and release occurred about 0.44 ms apart within a single 10 ms control interval. A later Recovery command cannot change the velocity already recorded at release. On recorded PPO01 observations from the 256-case training archive, a CPU replay of the native validation policies produced exactly equal pre-gate requests for PPO01 and the moderate candidate in all 35,708 primary and 35,289 companion frames. This isolates the policy function on common observations. It does not establish identical PhysX state histories for the four independent lane trajectories. A changed post-release arm or leg command has not been established as the cause of the lost ball.

The first fixed128 run after the rejection scores the retained PPO01. A separate, GPU-guarded fixed128 diagnostic has been scheduled for the moderate candidate to measure transfer without granting promotion. Independent512 remains reserved for final evaluation.
