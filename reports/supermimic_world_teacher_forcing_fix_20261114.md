# Structured world-model fit: teacher-forced window fix

The previous 32-step fit fed each predicted state back into the next supervised loss term. Before any actor update, this caused the world loss to rise from about `0.03` to `1.2+` on the native stream. The model was therefore unstable before it could provide a useful imagination target.

The fit now keeps the same 32-transition window and same reward/terminal losses, but uses the recorded next state as the next input between supervised terms. The imagination rollout remains fully autoregressive. This separates one-step physical fitting from model-based rollout stability without changing the task, action contract, physics, reward, or timebase.

CPU verification:

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.57s
```

The associated native run was stopped before any actor update; it is retained as a diagnostic run and is not a performance result.

