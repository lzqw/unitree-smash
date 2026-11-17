# Commands and exit status

All commands ran from the independent teacher worktree with existing `scripts/python.sh` / `/home/autodrive/Isaac/env_isaaclab/bin/python`.

Final regression (exit 0; 122 tests):
```
bash reports/impact_rate_validation_20261118/run_cpu_tests.sh
```

Fixed CPU replay (exit 0; no new native simulation):
```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh \
 -m scripts.replay_contact_candidate_gate_v1 \
 --anchored --reachable-impact --velocity-aware --impact-feasible-rate \
 --replanning --joint-stops --prediction-substeps 160 \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --interface reports/codex_teacher_joint_eval_20261116/controller.json \
 --output reports/impact_rate_validation_20261118/REPLAY.json
```
Raw output subsequently compressed losslessly. No rerun after diagnostic-only counters were added. `git diff --check`: exit 0.

Postprocessing first used system `python3`, failed with exit 1 (`ModuleNotFoundError: numpy`); reran with the existing Isaac interpreter, exit 0. No dependency installation or training environment change.
