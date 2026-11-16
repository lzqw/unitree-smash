# Precheck results — teacher_joint_eval_20261111_01

## Passed

- Worktree and HEAD captured in `audit/workspace_baseline.txt`.
- Python/dependency import paths captured in `audit/imports_and_deps.txt`.
- Direct-search source compiles with IsaacLab Python 3.11.
- New deterministic direct-search contract tests plus existing sequence-repair,
  native-action and AdmittedArmSupportQP tests: 17 passed.
- `git diff --check` clean.
- Normal-path class and original 32-node/16-tick constants retained.
- Candidate manifest written before any native launch.

## Native gate

The only native command for this candidate is the teacher-only entry with
`--teacher-only`, `--seed 882072`, 128 environments, 800 control ticks and
`--device cuda:0`. It is not launched while the unrelated
`supermimic_v1_native_eval_20261111` process owns GPU0.

The candidate must fail closed if the teacher-only flag is missing, a learned
recovery head/correction/expert bank is supplied, or the entry/source hashes
drift. No second native candidate is permitted.

## Not run yet

- 128-case native teacher-only evaluation: waiting for the unrelated GPU job to
  finish; no native result is claimed.
- Formal acceptance: false by contract.
