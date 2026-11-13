# Actor warmup diagnosis and fix

## Observed failure

Run `supermimic_structured_anchor_w0p32_i0p96_20261114_0228` was stopped at iteration 232 after the first stability failure:

- native reward remained approximately `-0.0115`;
- no native episode had completed, so no task event had yet been observed;
- world-model loss rose from approximately `0.03` to `0.23`;
- policy-anchor penalty reached approximately `7–15` during the late window.

The learner was performing imagined actor/critic updates as soon as the replay reached the batch size (roughly iteration 4), while each native lane had only a partial 8-second episode. Thus the actor was being optimized through a world model that had not yet seen a real terminal transition or task event.

## Change

`train_supermimic_horizon_v2.py` now accepts `--actor-warmup-episodes` (default `1`). Native collection and world-model fitting continue, but imagined actor/critic updates are withheld until the configured number of complete native episodes has been recorded. The default therefore waits for the first genuine 8-second episode boundary before changing the actor or critic. No action mapping, physics, reward definition, or event criterion changed.

Each history row records the completed-episode count, required count, and whether actor updates were allowed.

## Verification

```text
tests/test_supermimic_horizon_v2.py: 35 passed in 1.57s
```

The stopped run remains preserved as a diagnostic artifact. It was not relabeled as a task result and was not resumed with the unstable actor state.

