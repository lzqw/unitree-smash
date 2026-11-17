# Final-code reproduction

Use existing teacher worktree and `bash scripts/python.sh`. All runs used OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2. Use fresh output paths.

Final one-window CPU rollout, exit0:
```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.rollout_push_off_prefix_v1 --output reports/push_off_task_20261118/PREFIX_IMPLICIT.json
```

Final tests, exit0,25 passed:
```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_push_off_support_qp_v1.py tests/test_implicit_command_support_qp_v1.py tests/test_admitted_arm_support_qp_v1.py tests/test_action_conditioned_base_v1.py tests/test_contact_reach_bound_v1.py
```

One-state causal request/zero-increment audit entry:
```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.check_push_off_task_v1 --output NEW_OUTPUT.json
```
DIAGNOSED.json and RESULTS.json were run with INITIAL_INSTANTANEOUS_SOURCE.py.txt before the implicit mapping correction, not with final code. Both exit0; QP rejected. PREFIX and PREFIX_STRICT likewise preserve those failed proposals/fallback. Reproducing that historical formulation requires a separate isolated checkout of the archived class, never overwriting live files.

No native or training commands were run.
