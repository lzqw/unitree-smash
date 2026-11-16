# Recovery Arm WBC Round Audit

- `evaluation_stage`: `teacher_debug`
- `formal_eligible`: `false`
- `formal_acceptance`: `false`
- `formal_cases_read`: `false`
- Worktree: `/home/autodrive/LZQW/unitree_smash_teacher_eval`
- Branch/HEAD: `codex/recovery-arm-wbc-next-round-20261116` / `329c15e8977f0eb8588d533c225d5e3f713c5ac5`
- Frozen candidate run: `attempts/recovery_arm_wbc_next_round_20261116_03`
- Native service: `unitree-teacher-recovery-arm-wbc-20261116-03.service`
- Native result: 128 environments, 800 control ticks, 8 seconds, seed 882072, CUDA Isaac run, exit code 0, no signal, no stop request.

## Control chain evidence

The evaluator is teacher-only. `metrics.json` records `base_actor=false`, `approach_head=false`, `recovery_head=false`, `actor_forward_count=0`, and `optimizer_step_count=0`. The analytic DTG/Approach reference is used for command generation; the frozen checkpoint is loaded only for the existing evaluator contract and contributes no learned action. The Recovery path owns all 18 final commands after takeover (`all18_owner=true`, `old_residual_after_takeover=false`).

The actual command path was checked from the saved `pure_control_diagnostics.jsonl`, `pure_control_commands.npz`, and `sequence_dispatch.npz`: preview uses the live `previous` final command, dispatch round-trip is within `1.79e-7 rad`, and no row exceeds the `2e-6 rad` audit tolerance.

## Changes relative to `f9646c6`

1. Recovery prediction/search binds finite `state.previous_command_target` into the isolated interface before bounds, prediction, and direct search.
2. Teacher prediction and native dispatch both use the full-authority arbiter; the extra Recovery arm software slew is absent while native simulator limits remain.
3. Teacher-only analytic arm target slew is disabled; the six-axis measured Jacobian/racket reference path is retained.
4. Adopted direct-search arrays are normalized to list nodes before entering the legacy plan cache.
5. CPU contract tests cover history binding, physical admitted bounds, and no software arm slew.

## Independent work and SuperMimic

No SuperMimic source, checkpoint, worktree, process, or weight was modified by this round. The candidate manifest records its start-time process/cgroup snapshot. At final inspection the earlier SuperMimic PIDs were absent and only the TensorBoard tail remained; the launcher/cycle logs stopped at 01:39. No termination signal was issued by this run, so the later external termination cause is `UNKNOWN` and is not attributed to the teacher evaluation.

Pre-existing untracked files in the teacher worktree (`assets/robots/piper_standard/configuration`, `attempts/`, `audit/`) were not changed or added.
