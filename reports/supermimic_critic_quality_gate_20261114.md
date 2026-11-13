# Critic-quality gate

The ninth run showed that 200 critic optimizer updates were not enough: the previous critic loss was still roughly `466` when actor BPTT updates began, and native reward degraded again. The actor gate now requires both:

- the configured critic update count (`200` by default), and
- the previous critic loss to be at most `50` (`--actor-max-critic-loss`).

Until both conditions hold, imagination remains critic-only and the native initialized policy is preserved. The learner records the threshold and gate reason in each training row. The next run will use critic learning rate `1e-4`, actor lr `1e-5`, anchor coefficient `0.2`, and the same native/world-model settings.

CPU verification:

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.57s
```

