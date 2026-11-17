# Reproduction and observed exit status

Existing interpreter: `/home/autodrive/Isaac/env_isaaclab/bin/python`. Existing independent teacher worktree; no dependency replacement.

Final CPU tests (131 passed; exit0):
```bash
bash reports/task_constrained_hit_20261118/run_cpu_tests.sh
```

Single fixed CPU replay (exit0; output later compressed losslessly):
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh \
 -m scripts.replay_contact_candidate_gate_v1 \
 --anchored --task-constrained-hit --replanning --joint-stops --prediction-substeps 160 \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --interface reports/codex_teacher_joint_eval_20261116/controller.json \
 --output reports/task_constrained_hit_20261118/REPLAY.json
```
The old `--impact-feasible-rate` flag is deliberately absent and incompatible with v2. The old `--reachable-impact` soft objective is superseded in v2. Original24 event candidates and fixed causal inputs retained.

Native historical calibration: exact child command is `attempts/calibration01/command.json`, source hashes in MANIFEST.json, execution identity/cgroup in identity.json, terminal exit in exit.json. Launched using:
```bash
systemd-run --user --unit=unitree-teacher-calibration-20261118-01 \
 --slice=unitree-teacher.slice --property=Restart=no \
 --property=RuntimeMaxSec=infinity --property=TimeoutStartSec=infinity --property=TimeoutStopSec=infinity \
 --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
 /usr/bin/python3 /home/autodrive/LZQW/dual_line_runtime/bin/line_supervisor.py \
 --line teacher --run-id=calibration-20261118-01 \
 --output=/home/autodrive/LZQW/unitree_smash_teacher_eval/reports/task_constrained_hit_20261118/attempts/calibration01 \
 --cwd=/home/autodrive/LZQW/unitree_smash_teacher_eval \
 --command-json=/home/autodrive/LZQW/unitree_smash_teacher_eval/reports/task_constrained_hit_20261118/attempts/calibration01/command.json
```
A reproduction must use a new run_id/output directory. No automatic restart or policy evaluation is involved.

Post-native model comparison:
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh \
 -m scripts.analyze_historical_teacher_calibration_v2 \
 --native reports/task_constrained_hit_20261118/attempts/calibration01/native \
 --interface reports/codex_teacher_joint_eval_20261116/controller.json \
 --output reports/task_constrained_hit_20261118/CPU_PHYSX_COMPARISON.json
```

Initial new-test log contains a genuine test failure (explicit configuration validation was not invoked). It is preserved. The initial compound shell also printed the log, so its shell exit0 is not presented as a passing pytest result. Corrected and full-suite outputs are separate.
