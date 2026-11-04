# Causal expert-bank distillation: native result

## Decision

The offline distilled policy is rejected as the current best policy. It fits the recorded expert actions closely, but native closed-loop evaluation gives **21/128 complete5** on both fixed seeds. The frozen approach policy remains **27/128**, while the explicit causal expert-bank controller trial remains **28/128** on both seeds.

## What was run

The distiller in [`scripts/distill_case_expert_selector_v1.py`](../scripts/distill_case_expert_selector_v1.py) copied the frozen approach policy and trained a small public-observation residual head. After the first causal ball observation, the training target was the measured expert selected by the case bank; before that point the target stayed at the candidate action. It ran 1,800 CPU updates over 96,562 training and 96,562 validation samples. The exported checkpoint was explicitly marked `native_execution=false` until native evaluation.

The resulting checkpoint was then evaluated in native Isaac on the original 128 TRAIN cases, with the same 5 m/s complete5 metric, persistent recovery assistance, ground extent, actuator path, and precision receipt. Both fixed seeds completed with zero numerical failures.

| policy | complete5 | air contacts | recovery | actual racket contact | crossed net | target landings |
|---|---:|---:|---:|---:|---:|---:|
| frozen approach candidate | 27/128 | 74 | 34 | 110 | 71 | 64 |
| causal expert-bank controller trial | 28/128 | 76 | 38 | 108 | 75 | 71 |
| stateless distilled head, seed 882072 | 21/128 | 74 | 32 | 106 | 74 | 65 |
| stateless distilled head, seed 882075 | 21/128 | 74 | 32 | 106 | 74 | 65 |

The distilled policy retained 16 of the candidate's 27 complete cases, lost 11, and gained 5 on seed 882072. Thus the small action regression error did not imply trajectory equivalence. The likely limitation is structural: the expert-bank controller remembers the first causal ball feature and holds a case-conditioned expert, whereas the exported head only receives the current public observation and cannot reproduce that memory-dependent routing reliably.

A follow-up head held the first 18-dimensional causal ball feature per episode and was evaluated natively on seed 882072. It reached **17/128** complete5 (73 air contacts, 28 recoveries, zero numerical failures), so adding this memory input to a static corrective head still disturbed valid trajectories. The follow-up is recorded in [`memory_conditioned_distillation_result_20261104.json`](memory_conditioned_distillation_result_20261104.json) and is also rejected.

## Interpretation and next action

This is a useful negative result. It rules out promoting stateless offline action-MSE distillation as the next best policy and shows why the controller trial should not be described as pure RL. The next implementation should preserve causal memory or learn corrections online (for example, a phase-gated recurrent/conditioning module or DAgger-style corrective updates), then validate the resulting policy natively before promotion. The rejected checkpoint and both native metrics remain available locally for reproduction; only the compact report and source are committed here.

The experiment is development evidence on fixed TRAIN128. It retains the base controller and persistent recovery assistance, is not a formal acceptance score, and provides no unseen-condition generalization claim.
