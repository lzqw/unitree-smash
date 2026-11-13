# Warmup fallback gate

Deterministic native collection preserved the initialized policy but still produced early termination before any 8-second timeout. A strict full-timeout gate would therefore leave actor/critic permanently frozen.

The learner now has two explicit gates:

1. preferred: at least one real timeout episode;
2. fallback: at least eight terminated native episodes and `world_loss <= 0.75`.

The history records the gate reason (`full_timeout`, `terminated_episode_fallback`, or `waiting`), full timeouts, terminated episodes, and the world-loss threshold. This does not treat terminated episodes as successful episodes; it only allows the learner to begin conservative recovery from real failure data when no timeout is available.

The next run will use actor learning rate `1e-5` and anchor coefficient `0.2` to limit early policy drift. World-model fitting remains teacher-forced, `world_lr=5e-5`, one update per native iteration, and the task/timebase/physics contract is unchanged.

CPU verification:

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.57s
```

