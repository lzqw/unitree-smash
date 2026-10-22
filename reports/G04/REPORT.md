# G04 — FAILED

The independent native evaluation measured 400 ground-initialized episodes. Panel success fractions: {"dev_jump_v2_a": 0.0, "dev_jump_v2_b": 0.0}; the required minimum-panel score is 0.0000 (threshold 0.90). Numerical failures: 0. This is a development evaluation, not the unopened G10 holdout. Engineering execution and PPO updates alone do not establish task success.

Frozen evaluation protocol: configs/evaluation_protocol_v2.json. The entire robot's mass-weighted COM must rise 0.08 m after actual takeoff in the same flight that has at least 0.10 s continuous all-feet clearance. After landing, posture, horizontal/root angular/joint velocity and non-foot-ground contact constraints must hold for a continuous 0.80 s. Metrics are sampled every physics substep; body, arm, leg non-foot links, mounting plate and racket ground contacts invalidate success. The two panels contain exactly 200 unique scenarios each; initial physical states are retained in initial_states.csv. Deterministic PPO inference uses no assisting controller.

Observed failure counts (overlapping): {"body_contact": 157, "no_qualified_jump_landing": 400, "no_contiguous_recovery": 400}. Qualified jump landing fraction: 0.0000; maximum observed COM rise: 0.00958 m. These results support neither a claim of trained smashing nor a proof that the hardware is incapable.

Audited standard PiPER and full payload remain in assets/quadsmash_standard_v2.usd, total mass about 20.6469917 kg; original geometry and actuator limits retained. Separate bounded-action engineering search reports/G04/capability_all_body_v2 found 17 qualified jump landings in 256 trajectories but zero complete recovery successes. That search is not a learned policy result. Earlier three training revisions used invalid physics and remain archived with their costs. The first corrected baseline also omitted the specified unsafe-body-contact reward cost; it was safely stopped, retained, and explicitly corrected in jump_ppo_v2b.json. No acceptance threshold was reduced.

Checkpoint: artifacts/G04/policy.pt, SHA256 0c94cecf197b9dd5ab658fe391e42122fbbefdc8e10227d90c4f05f8ab26818f. Source evaluation: reports/G04/corrected_v2c_continue_03_eval; training run: runs/G04/corrected_v2c_continue_03. Each native job retains command, source snapshot/hashes, logs, seeds and receipt. Full resumable checkpoint.pt files begin at a canonical ground episode boundary after a completed PPO update, preserving optimizer/normalizers/RNG/counters; arbitrary in-contact PhysX replay is not claimed.

G04 cumulative charged GPU0 walltime: 3.432440 h / 8 h. Conservatively charged all-environment interactions: 99,685,408 / 200,000,000. This includes failed attempts, diagnostics, evaluation and a historical reserve of 2 h / 25 M interactions; the reserve is not a measured historical timing claim. G05+ is not authorized.

Evaluation configuration: /home/autodrive/LZQW/unitree_smash/configs/jump_ppo_v2c.json. Subsequent shaping changes and their hypotheses are retained in status/DECISIONS.md and each run's source snapshot. Raw action saturation counters from runs before wrapper clipping was removed (including corrected_v2b_main01 training) are invalid; actual actuator torque/speed records are retained separately.

Reproduce: python3 scripts/evaluate.py --stage jump_recover --checkpoint artifacts/G04/policy.pt --config /home/autodrive/LZQW/unitree_smash/configs/jump_ppo_v2c.json --suite dev_jump_v2 --episodes 400 --num-envs 200 --reset-mode ground --output reports/G04/NEW_EVALUATION_DIRECTORY
Verification: python3 tools/verify_goal.py --root . --goal G04 (expected to fail when measured performance is below threshold).

## Independent development evaluations

| Training run | Minimum panel success | Non-foot contact | Learned noise std |
|---|---:|---:|---:|
| corrected_v2_smoke | 0.0% | 21.0% | not logged |
| corrected_v2b_main01 | 0.0% | 100.0% | 0.7032 |
| corrected_v2c_main01 | 0.0% | 0.0% | 0.7477 |
| corrected_v2c_continue_01 | 0.0% | 100.0% | 0.9021 |
| corrected_v2c_continue_02 | 0.0% | 100.0% | 1.1370 |
| corrected_v2c_continue_03 | 0.0% | 39.2% | 1.4932 |

Each row is a separate 400-episode development evaluation. Sparse height/flight events with body contact or no recovery remain failures. Growing policy noise and the gap between stochastic training and deterministic evaluation motivate the proposed low-noise test; they do not prove an exclusive cause. The proposal is not an observed improvement and requires the explicit extra-revision authorization described in additional_revision_proposal.json.

## Supplementary foot-contact audit

In 256,000 environment physics substeps, the production net foot-force sensor and four independent ground-filtered foot sensors had maximum force-vector discrepancy 0 N and 0 airborne-label disagreements. Controls included zero-action standing, replayed existing bounded-action trajectories, deterministic policy inference and stochastic policy inference. This finite diagnostic does not prove agreement in all possible configurations; no production sensor or threshold was changed.

Replay of selected existing engineering trajectories yielded 30/32 qualified jump landings and 1/32 complete recoveries. These curated controller trajectories are not a learned-policy evaluation and do not change the failed independent 400-episode PPO result. Source: reports/G04/foot_ground_audit_v1.
