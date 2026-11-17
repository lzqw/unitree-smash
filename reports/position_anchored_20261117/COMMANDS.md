# Commands and exit status

All commands below executed in `/home/autodrive/LZQW/unitree_smash_teacher_eval` using the existing interpreter `/home/autodrive/Isaac/env_isaaclab/bin/python` via `scripts/python.sh`. OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2 for CPU runs. No Isaac native launch command executed. JSON replay outputs were losslessly gzip compressed after completion where indicated.

| Entry | Output | Exit |
|---|---|---:|
| `bash reports/position_anchored_20261117/run_cpu_tests.sh` | CPU_DELIVERY.txt, CPU_EXIT.json | 0 (89 passed) |
| `-m scripts.compare_position_anchor_history_v1` | SAME_INPUT_FINAL.json, SAME_INPUT_FINAL_LOG.txt | 0 |
| `-m scripts.replay_contact_candidate_gate_v1 --anchored --replanning` | REPLAY_DELIVERY.json.gz, REPLAY_DELIVERY_LOG.txt | 0 (diagnostic completes; candidate gate FAIL) |
| `-m scripts.pair_rr_calf_replay_v1` | RR_PAIRED.json, RR_PAIRED_LOG.txt | 0 (all physics prefixes rejected) |
| same with `--joint-stops` | RR_STOP_TRACE.json.gz, RR_STOP_TRACE_LOG.txt | 0 (all fail solver at node9) |
| `git diff --check` | no output | 0 |

Common input arguments for replay/comparison:

```bash
--input reports/joint_loop_round_20261117/attempts/probe01/native \
--checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
--output <new-output-path.json>
```

Candidate gate additionally uses:

```bash
--interface reports/codex_teacher_joint_eval_20261116/controller.json
```

Paired replay additionally uses:

```bash
--prior-replay reports/reachable_replan_20261117/REPLAY_DELIVERY.json.gz
```

Scripts refuse to overwrite output. Large checkpoint and native recording remain existing local experiment inputs; their SHA256 are in GATES.json. A Git checkout without those local inputs cannot rerun the historical data experiment; do not invent replacements. The old git blob is loaded only by the offline same-input comparison, never the online controller.

Artifact lint: first cached diff check exited2 because a raw patch stored as text has blank context lines consisting of one space. Patch was preserved losslessly as SOURCE_DIFF.patch.gz; source code whitespace was not the cause. Final cached check is recorded separately.
