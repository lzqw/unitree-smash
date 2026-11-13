# SuperMimic structured-history + actor-anchor run

Date: 2026-11-14 (Asia/Shanghai)

## Purpose

This run tests a structured-history world model on the full 794-dimensional task observation while keeping the initialized PPO policy close to its verified action mapping. It is a training run, not a formal 128-environment evaluation.

## Code and provenance

- Branch: `supermimic-structured-history-event-metrics-20261114`
- Latest pushed commit: `7807811`
- Initial policy: `artifacts/ppo_full_racket_residual_v4/policy.pt`
- Environment: 512 parallel Isaac environments, seed `20261114`
- Control period: `0.01 s`
- World-model window: `0.32 s` (`32` transitions)
- Imagination horizon: `0.96 s` (`96` control steps)
- Initial actor learning rate: `5e-5`
- World/critic learning rates: `2e-4`
- Policy anchor coefficient: `0.05`
- No MPC, teacher takeover, BC, or external driver is used.

## Runtime snapshot

At iteration `117`:

| Metric | Value |
|---|---:|
| Native steps | 59,904 |
| Replay size | 59,904 |
| World loss | 0.04658 |
| Imagined return | 2.0861 |
| Actor gradient norm | 5.3066 |
| Policy-anchor penalty | 0.9888 |
| Native reward mean | -0.01203 |
| Native positive-reward fraction | 0.0 |
| Native done count | 0 |
| GPU peak allocated | 5.04 GB |

Per-iteration time is approximately `8 s`; the process is using CUDA and remains active.

## Event status

All native task event counters are still zero at this snapshot. This is expected at this stage: each lane has advanced only about `1.17 s`, while the configured launch/impact events occur later in the episode. These zeros are therefore not a success-rate measurement. The first meaningful native event check is expected after roughly `200–300` iterations; a complete 8-second episode requires substantially more iterations.

The native event logger records episodes, qualified airborne string contact, ball-side completion excluding recovery, recovery success, `complete5`, frame hits, non-foot contacts, body-ground observations, and rally faults before reset-state overwrite.

## Interpretation

The structured model is numerically stable so far, and the actor anchor is active. Imagined return alone is not evidence of task learning; native reward and event metrics are the acceptance signals. The run will remain active until native events are observed or a real degradation/failure is detected.

## Reproducibility

```bash
/home/autodrive/Isaac/env_isaaclab/bin/python scripts/train_supermimic_horizon_v2.py \
  --headless --device cuda:0 \
  --output /home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline/reports/supermimic_structured_anchor_w0p32_i0p96_20261114_0228 \
  --init-policy-checkpoint /home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline/artifacts/ppo_full_racket_residual_v4/policy.pt \
  --iterations 5000 --num-envs 512 --seed 20261114 \
  --world-batch 2048 --world-updates 3 --imagination-batch 2048 \
  --actor-microbatch 128 --actor-lr 5e-5 --world-lr 2e-4 --critic-lr 2e-4 \
  --policy-anchor-coeff 0.05 --replay-capacity 250000 --checkpoint-every 500 \
  --persist-replay --world-window-seconds 0.32 --imagination-seconds 0.96
```

