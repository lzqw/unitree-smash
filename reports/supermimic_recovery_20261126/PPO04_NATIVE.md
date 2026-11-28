# PPO04 native Recovery implementation

The incumbent remains PPO01. Its fixed128 result was 10 strict recoveries, 25 relaxed recoveries and 71 body-ground cases among 108 qualified contacts. It achieved 96 ball-side completions and zero robot-net faults in 128 cases.

The optional macro mode trains on all 256 training cases. After the release protection interval, it samples a bounded request every four control steps. Between decisions it holds the offset relative to a frozen state-dependent policy. The action likelihood uses Gaussian tail probabilities at the stops. Returns include intervening rewards and partial terminal intervals. Deployment still uses the deterministic head at each control step. Native evaluation must measure the effect of this train/deploy difference.

The lift reward uses a broad height slope, reduces the incentive to keep the legs still at low height, and uses the lowest point of the authored RL thigh box. A failed low-root termination receives a terminal penalty on the transition before reset. Training keeps the full episode for late falls.

A separate deterministic dev128 run with the primary Approach establishes the baseline before training. The runner requires at least eight strict baseline successes. It compares case-matched results every three Actor updates, with a final fixed128 evaluation for candidates passing the development check. The selected development panel has 128 distinct launches and no case identifier or launch overlap with train256_r13_ab, fixed128 or independent512. Its strict-success coverage still requires native measurement.

CPU regression tests passed, including the censored likelihood, truncated four-step returns and rotated collision box geometry. No native result is claimed for PPO04.
