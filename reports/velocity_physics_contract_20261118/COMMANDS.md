# Executed commands / results

CWD `/home/autodrive/LZQW/unitree_smash_teacher_eval`; existing `bash scripts/python.sh` interpreter; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2. No packages installed, no GPU/native service started.

- `bash reports/velocity_physics_contract_20261118/run_cpu_tests.sh`: exit0,109 passed. Three logs preserve integration/metadata validation stages; final ALL_TESTS_DELIVERY.txt.
- New velocity/bundle tests: exit0,7 passed.
- `git diff --check`: exit0.

Fixed CPU replay, exit0 (candidate acceptance still FAIL):

```bash
bash scripts/python.sh -m scripts.replay_contact_candidate_gate_v1 \
 --anchored --reachable-impact --velocity-aware --replanning \
 --joint-stops --prediction-substeps 160 \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --interface reports/codex_teacher_joint_eval_20261116/controller.json \
 --output <new-output.json>
```

REPLAY and REPLAY_AUDITED are the same fixed algorithm/configuration, latter with additional metadata/export logging. No native selection or parameter sweep. A failed audit launch mistakenly supplied `reports/codex_teacher_joint_eval_v1.json`; exit1/FileNotFoundError before planning is preserved in REPLAY_DELIVERY.log. Corrected path used for REPLAY_AUDITED, exit0. Final original-target metadata whitelist fix was checked by production-branch regression, and old errors reconstructed only as posthoc diagnostic in VELOCITY_REQUEST_AUDIT.json; no re-optimization from those results.

Paired160substep CPU diagnostic, exit0 but solver fails all3:

```bash
bash scripts/python.sh -m scripts.pair_rr_calf_replay_v1 --joint-stops --substeps 160 \
 --input reports/joint_loop_round_20261117/attempts/probe01/native \
 --checkpoint reports/joint_loop_round_20261117/attempts/probe01/candidate.pt \
 --prior-replay reports/reachable_replan_20261117/REPLAY_DELIVERY.json.gz \
 --output <new-output.json>
```

Large JSONs losslessly gzip-compressed after completion. Scripts refuse overwrite. Original local checkpoint/observations required; hashes retained in replay/RESULTS.json. Do not replace missing inputs with synthetic data. PhysX validation and complete8-second scoring NOT_RUN; no valid reference archive produced.
