# Reproduction / exit codes

Existing interpreter: `/home/autodrive/Isaac/env_isaaclab/bin/python` via `bash scripts/python.sh`. Working directory `/home/autodrive/LZQW/unitree_smash_teacher_eval`. All CPU runs set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2. No dependency installation or native/GPU job.

- `bash reports/normal_slip_validation_20261117/run_cpu_tests.sh`: exit0;101 passed.
- Requested six-file pytest: exit0;74 tests before local edits,77 after; logs retained separately.
- `git diff --check`: exit0.

All following script processes exited0 because diagnostics were successfully saved. **Candidate/physics gate failures in those JSONs are still failures, not passing controllers.** Commands use `bash scripts/python.sh -m`:

```bash
scripts.replay_contact_candidate_gate_v1 --anchored --replanning \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --interface reports/codex_teacher_joint_eval_20261116/controller.json \
 --output <unique-output.json>

scripts.pair_rr_calf_replay_v1 --joint-stops \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --prior-replay reports/reachable_replan_20261117/REPLAY_DELIVERY.json.gz \
 --output <unique-output.json>

scripts.compare_position_anchor_history_v1 \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --output <unique-output.json>
```

Each script refuses output overwrite. Saved large JSONs were losslessly gzip compressed after execution; logs retain original filenames. Checkpoint/native recordings remain existing local inputs with hashes in replay. Raw data must not be replaced with invented substitutes on another machine. No native run is implied by these commands.
