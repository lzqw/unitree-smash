# Reproduction (all commands use the existing teacher environment)

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.refine_action_conditioned_contact_v1 --replay reports/experimental_fullbody_gate_20261118/REPLAY.json.gz --observations reports/joint_loop_round_20261117/attempts/probe01/native/public_observations.npy --output reports/action_conditioned_contact_20261118/RESULTS_WITH_BOUND.json
```
Exit0. Use a new output path; prior outputs are protected. The first run before adding reach-bound diagnostics is retained as RESULTS.json, also exit0.

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_action_conditioned_base_v1.py tests/test_contact_reach_bound_v1.py tests/test_experimental_sphere_admission_v1.py tests/test_contact_first_fullbody_evaluator_v1.py tests/test_offline_reference_bundle_v1.py tests/test_joint_prefix_execution_v1.py tests/test_task_constrained_hit_v2.py tests/test_position_anchored_contact_search_v1.py tests/test_contact_first_trajectory_search_v1.py
```
Exit0:125 passed. No native or training launch commands.
