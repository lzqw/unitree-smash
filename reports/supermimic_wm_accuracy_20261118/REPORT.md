# SuperMimic WM accuracy checkpoint

## Scope

This report records the bounded action-conditioned ball-response calibration and
the current gate decision. It does not report Actor training, a native task
evaluation, or a successful contact-validation receipt.

## Current code

- Branch: `feat/supermimic-native-physical-loop-20261114`
- Commit: `3db6e32c5df2d98026255f7491437af48fdafd0b`
- The calibration script now evaluates held-out native ball-response rows every
  16 updates and restores the best validation state before writing a candidate.
- CPU checks after the change: 16 passed. The broader previously recorded
  targeted set was 37 passed.

## Latest candidate

Output directory:

`/home/autodrive/LZQW/experiments/supermimic_ball_response_fit_multi_best_20261118_203802413079814`

| Artifact | SHA256 |
|---|---|
| `world.pt` | `a2e6609ded2d4819ba180f4f02897d0f8e1cf225e0ce5f14237fe47f78560a7f` |
| `action_ball_candidate.pt` | `bd1bad50f3f9cabeefd66ab0dca7eb7cedd4c954dbeea9f832f0d59d8cab478d` |
| `completed.json` | `893469fe5bb094327cae93221fb715257456f87d1ce2ab4f8e44a4ea4e523185` |

The run performed 256 bounded world-response updates, with zero Actor and zero
Critic updates. Peak recorded GPU allocation was about 1.41 GB. The completion
receipt is `FIT_COMPLETED_NOT_ADMITTED` and `native_admission` is false.

## Held-out result

The held-out validation split contains 23 nonzero-event rows for each listed
event offset.

| Offset | Initial velocity RMSE | Candidate velocity RMSE | Initial forward-sign error | Candidate forward-sign error |
|---:|---:|---:|---:|---:|
| -1 | 5.1692 | 5.3456 | 47.8% | 60.9% |
| 0 | 1.7653 | 2.1684 | 26.1% | 21.7% |
| +1 | 0.08175 | 0.08178 | 0% | 0% |

The candidate improves the offset-0 sign error slightly but worsens the
contact-edge velocity RMSE and the pre-contact metrics. It is therefore not
eligible for Actor contact BPTT or promotion.

The native paired-response bundle has SHA256
`04957798ee77a294ea969fbab405fe6cb8e8bf5033b984ae8a29fb3a2b5c6dad`. Its
receipt remains uncalibrated, so it is diagnostic evidence only.

## Runtime status

No large-scale training process is running. At report time the Isaac Python
environment reported `torch.cuda.is_available() == False` and NVML could not
initialize. The candidate calibration was completed before this driver state
was observed. A GPU/driver recovery is required before another GPU fit or
large-scale training run.

The original `/tmp` source archives referenced by the calibration command are
not currently present. Their hashes and provenance remain recorded in the
candidate `config.json` and completion metadata. No missing source is replaced
with a fabricated transition.

## Decision

Keep the candidate as a diagnostic artifact. Do not load it for policy updates,
do not start the 256-update Actor run, and do not claim improved native task
return. The next admissible step is a new calibration using restored native
source archives and a working GPU, followed by an independent native response
check.
