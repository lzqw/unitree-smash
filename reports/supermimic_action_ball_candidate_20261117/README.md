# SuperMimic same-step action-conditioned shuttle response: experimental patch

## Scope / provenance
- Base: `dc1d6f1a9c40a11b813c0f32122fce4ae7c008cd`.
- Dedicated review branch: `feat/supermimic-action-conditioned-ball-20261117`.
- No native weights, physical assets, real checkpoint, task reward, 100 Hz controller, or G07 mixer changed.
- The preceding frozen TRAIN128 result (74 qualified contacts, 54 ball-side, 26 complete5, return -10.87400572) remains **unmodified**. It is **not** a learning result.

## Root causes identified in source
1. Legacy `ball_transition` feeds the ball correction exclusively with CURRENT racket/ball geometry; the next ball state has no direct dependency on this tick's final joint target. Fixed-state one-step ball/action derivative is structurally zero.
2. `PhysicalHorizonTrainer.fit_world_window` measures `ball_i` but does not independently include it in the WM objective. The legacy physical 55D state loss DOES supervise ball kinematics, so it is inaccurate to claim there was no ball supervision.
3. The reported 7.37 m/s and 100% forward-sign error are at offset -1 relative to the native latch edge, not necessarily the exact impact transition. An early predicted bounce must be distinguished from missing action coupling; no root cause is certified without offset 0/+1 comparison and PhysX probe.

## Changes
- `supermimic_action_ball_v1.py`: explicit opt-in `ActionConditionedBallV1`. Uses only causal current physical state, this-step command-conditioned predicted robot state, free-flight prediction, and pinned racket geometry to calculate 0/mid/1-tick sweep gaps. Its trained MLP adds a smooth near-contact 6D ball correction. Output layer initializes to ZERO for baseline compatibility.
- `supermimic_ball_sequence_v1.py`: new keyword-only `action_model=None`; legacy output is unchanged without a model.
- `supermimic_physical_imagination_adapter_v1.py`, learner and validation: identical conditional correction in BPTT, recorded-command WM fit, and validation. With the candidate present ONLY, WM gains a separately weighted physical-unit ball loss (4x on native racket-edge transitions). No change to the baseline WM objective when disabled.
- `scripts/fit_supermimic_action_ball.py`: offline-only supervised candidate fit. Uses independent train/validation lane split, recorded final targets, 50% racket-edge positives, 25% preceding SAME-EPISODE hard negatives (avoid one-tick-early bounce), 25% other negatives. Reports one-step offset -1/0/+1 RMSE and forward sign, as well as candidate's model-only arm-target sensitivity.
- `supermimic_physical_loop_loading_v1.py`: optional source/asset/racket/net geometry hash-matched action model loading. No automatic promotion.
- Training launcher can pass `SM_ACTION_BALL_CHECKPOINT`; model is REFUSED before native WM/Actor training unless a calibrated current-version physical probe is provided for BOTH heads.
- `scripts/validate_supermimic_physical_wm.py`: optional candidate evaluation path. Runtime identity records the new code hash.

## Offline fit (not executed in this delivery)
```bash
python scripts/fit_supermimic_action_ball.py \
  --physical-checkpoint /path/to/physical.pt \
  --ball-checkpoint /path/to/ball.pt \
  --packed-data /path/to/real/packed/data.pt \
  --split-manifest /path/to/disjoint_lanes.json \
  --output /path/to/new/output \
  --device cuda:0 --updates 400 --batch 128
```
`disjoint_lanes.json` requires `train_lanes` and `validation_lanes`. The baseline physical/ball checkpoints and exact case order must match the native dataset. Dataset-level split alone does NOT prove the prior baseline WM never saw a validation case; inspect its original fit manifest.

The resulting `action_ball_candidate.pt` is a MODEL-ONLY candidate. Inspect `comparison.json` for offset -1, 0, +1; do not deploy because its gradient is nonzero. Explicitly reject the candidate if it reduces training loss but worsens heldout contact RMSE, off-contact dynamics, or action sensitivity under real physics.

## Validation performed now
- Four *isolated synthetic* CPU contracts passed under a stub package (geometry, legacy ball model and base robot model were simplified stand-ins). These demonstrate zero-initialized legacy behavior, a nonzero same-step ball/target Jacobian after fitting, absent-ball masking, and reset-safe hard negatives.
- Python syntax compiled for the new candidate module and offline fit script in the isolated workspace.
- Full project 253-test suite, native environment startup, GPU WM fit, paired PhysX action counterfactual, and actual R8/A8 optimization WERE NOT run with this patch.

## Remaining unimplemented / unverified milestones
1. Real-recorded-data candidate fitting and independent complete-episode validation for -1/0/+1, 1/4/16/32/64 steps; >16 early contacts must be audited for cohort bias.
2. Paired native PhysX ±delta command probes from truly matching execution and solver states (or a recognized repeatable same-prefix protocol). The true `d v_ball / d u_final` direction and scale are UNKNOWN.
3. Confirm learned arm target changes survive G07 final mixer, rate limiting, and 100 Hz physical execution; signed Jacobian alignment and command budget gate.
4. Independent WM fitting provenance for the holdout lanes; measure Recovery foot/contact model errors and both heads' value trust.
5. Real actor/world/critic optimizer updates through R8 -> A8 -> frozen combination -> R8; save/restore and formal TRAIN128 evaluation.
6. complete5 candidate must pass the original 27/128 requirement without changing the task metrics. Existing frozen causal-gate baseline is 26/128 and cannot be promoted.

**Status: candidate SOURCE-CODE path prepared, synthetic CPU evidence only. Not production-ready, no measured performance improvement.**
