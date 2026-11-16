# Teacher-only native evaluation report

- evaluation_stage: `teacher_debug`
- formal_eligible: `false`
- formal_acceptance: `false`
- formal_cases_read: `false`
- candidate: `teacher_joint_eval_20261111_02`
- run: `/home/autodrive/LZQW/unitree_smash_teacher_eval/attempts/teacher_joint_eval_20261111_01/native_teacher_joint_20261116_01/native`
- completed: `800/800` ticks and `128/128` first episodes
- process exit: `0`, normal, no stop request

## Results

- qualified airborne string contact: **29 / 128**
- Q (airborne + legal ball-side conditions): **21 / 128**
- episode recovery field: **21 / 128**
- H and recovery: **21 / 128**
- Q and recovery: **15 / 128**
- complete5: **10 / 128**
- conditional recovery reported by evaluator: **20 / 29**
- numerical failures: **0**
- action round-trip audit: max **0 rad**, 2e-6 check **PASS**

## Teacher and process audit

The evaluator recorded `actor_forward_count=0`, `optimizer_step_count=0`, and disabled base actor, approach head, and recovery head contributions. The native Recovery adapter recorded `all18_owner=true` and `old_residual_after_takeover=false`. The wrapper recorded distinct supervisor/child PGID and SID, a dedicated cgroup, `RuntimeMaxSec=infinity`, and normal exit.

The external SuperMimic process was not started or modified by Codex. Its separate attempt later failed in its own code with `KeyError: episode_index`; no restart was performed.

## Direct search result

The new direct search was attempted 333 times, evaluated 0 candidates, adopted **0** sequences, and found 0 feasible candidates. All attempts failed before candidate evaluation with `ValueError: Actual command history required`. Therefore the substantive sequence-search change was **not actually adopted** in this native run; the result is a teacher-only native run with the existing normal/repair/fallback behavior plus a failed direct-search entry path.

The raw evidence is in `TEACHER_DIAGNOSTICS.json`, `metrics.json`, `episodes.json`, `pure_control_diagnostics.jsonl`, `pure_control_commands.npz`, `sequence_dispatch.npz`, and `exit.json`.
