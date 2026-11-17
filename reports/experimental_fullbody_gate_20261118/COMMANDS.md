# Commands and observed exit codes

Working directory: `/home/autodrive/LZQW/unitree_smash_teacher_eval`.
Interpreter: existing `bash scripts/python.sh`; OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2.

Final adapter gates (exit0,65 passed):
```
bash scripts/python.sh -m pytest -q tests/test_experimental_sphere_admission_v1.py tests/test_contact_first_fullbody_evaluator_v1.py tests/test_offline_reference_bundle_v1.py tests/test_joint_prefix_execution_v1.py
```
Search regressions (exit0,48 passed):
```
bash scripts/python.sh -m pytest -q tests/test_task_constrained_hit_v2.py tests/test_position_anchored_contact_search_v1.py tests/test_contact_first_trajectory_search_v1.py
```
The initial erroneous test command referenced `test_task_constrained_hit_search_v2.py` and `test_task_constrained_impact_goal_v2.py` (nonexistent), exit4. No source fix was needed; command corrected, raw output preserved.

Single fixed replay (exit0):
```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m scripts.replay_contact_candidate_gate_v1 --anchored --task-constrained-hit --replanning --joint-stops --prediction-substeps 160 --experimental-spheres reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json --input reports/joint_loop_round_20261117/attempts/probe01/native --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt --interface reports/codex_teacher_joint_eval_20261116/controller.json --output reports/experimental_fullbody_gate_20261118/REPLAY.json
```
Use a new output path for a reproduction. Original raw JSON compressed losslessly after completion.

Read-only summaries and FK attribution (both exit0):
```
python3 reports/experimental_fullbody_gate_20261118/summarize.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m reports.experimental_fullbody_gate_20261118.analyze_terminal
```
No native launch command in this round. No EVALUATION_COMPLETE or valid-native reference marker.
