# Deterministic native warmup

The previous run used sampled actions from the first native step. With the initialized residual policy and `log_std` noise, lanes terminated before the 8-second timeout, so `full_episodes` stayed zero and actor updates never had a trustworthy full-episode gate. The world model then saw increasingly damaged states.

The native collector now uses deterministic actor means until at least one imagined actor update has occurred. Sampling resumes only after the actor is actually updated (or when resuming an already trained checkpoint). History reporting now distinguishes full timeouts from terminated episodes in `native_episode_warmup`.

No action decoding, physics, reward, success criteria, or timebase changed.

CPU verification:

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.59s
```

