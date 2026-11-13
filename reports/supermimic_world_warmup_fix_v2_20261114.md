# World-model warmup v2

## Second diagnostic

The first warmup patch counted every reset as a completed episode. In the next run, at iteration 320, `episodes=12` opened actor updates even though those were early failure resets. At the same time, world loss had increased to `1.33` while the actor was still close to its initial policy. This was not valid full-episode warmup.

## Fixes

- `full_racket_env_v4.py` now records `episode_timeout` and `episode_duration_s` before reset.
- Actor/critic warmup now counts only real episode timeouts (`full_episodes`), not early terminations.
- The next run should use `--world-lr 5e-5 --world-updates 1`; this is a stability adjustment for the world-model fitting stage only. Task actions, physics, reward definitions, and timebase remain unchanged.

## Verification

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.57s
```

The interrupted run with early-reset counting is preserved and is not used as a performance result.

