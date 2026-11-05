# Recovery-only causal expert teacher: result

Date: 2026-11-05

This run tested a causal top-5 historical expert bank as a recovery-only teacher while training a new approach head. The base actor, existing recovery head, actor normalizer, and log standard deviation were frozen. The teacher used blend `0.5` only in the configured recovery phase.

## Training

- Output: `/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/causal_expert_teacher_recoveryonly05_frozenv2_8_env5120_02`
- 5120 environments, 8 updates, seed `882144`
- Initialization: `frozen_v2_smooth_approach16_env5120_01/policy.pt` (current pure-student 27/128 baseline)
- Top-5 case expert bank, blend `0.5`, recovery-only gate
- Training episodes: 6,978
- Training-time air contacts: 3,510
- Training-time conditional recoveries: 1,138
- Training-time complete5 events: 710
- Numerical failures: 0

Training counts include teacher assistance and are not a frozen-policy acceptance result.

## Frozen evaluation with the extra teacher disabled

The exported policy was evaluated on all 128 fixed TRAIN cases with the same persistent base recovery assistance used by the baseline. Both seeds produced the same result:

| Seed | complete5 | airborne contacts | conditional recovery | crossed net | landed target | numerical failures |
|---:|---:|---:|---:|---:|---:|---:|
| 882072 | 24/128 | 73 | 34 | 71 | 64 | 0 |
| 882075 | 24/128 | 73 | 34 | 71 | 64 | 0 |

Current pure-student baseline `frozen_v2_smooth_approach16_env5120_01` is 27/128 on both fixed seeds. The recovery-only teacher candidate is therefore **rejected** and is not promoted as the new checkpoint.

## Interpretation

The teacher generated many assisted successes during training, but that gain did not transfer to the student when the teacher was removed. The unchanged air-contact and recovery counts, together with the lower complete5 result, indicate that this blend mainly supplied execution assistance rather than a recoverable policy improvement. Keep the 27/128 pure-student checkpoint as the active baseline and do not continue this blend scan.

Evaluation artifacts remain local; large checkpoints and videos are intentionally not committed.
