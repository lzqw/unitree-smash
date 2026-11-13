# PPO-initialized SuperMimic status — 2026-11-13

This branch now uses the measured full-racket residual PPO policy as a strict
SuperMimic actor initialization. The source policy is
`artifacts/ppo_full_racket_residual_v4/policy.pt` (SHA-256:
`772608a1c42ec78c09cdf3bd8b7bfe3fed1738d1273cde9d09610022553aa21b`).

The older `reports/G07/train_high_attack_all01/checkpoint.pt` was rejected for
this purpose: its zero-update full-racket evaluation produced no qualified
contact and 7041 numerical faults. It uses the older direct `jump-event-v1`
contract and does not represent the measured full-racket residual execution
policy. This is an initialization-contract failure, not evidence that
SuperMimic cannot learn the task.

## Code changes

- `ResidualPPOInitializedActor` loads the embedded TorchScript residual actor
  with strict 794-observation/18-action checks.
- `SuperMimicEnvV1` applies the original frozen analytic base plus the native
  residual mixer before the unchanged full-racket actuator path.
- The duplicate environment initializer/reset path was consolidated so the
  residual controller is actually installed and reset per environment.
- Training records the requested action, final action, and their max/mean
  difference each iteration. The action contract remains GPU-resident and
  uses the original physical limits and arm target-rate limiter.
- The learner keeps the 8 s/800 control-tick contract. The actor is held until
  102,400 native transitions (128 environments × 800 ticks), so the first
  actor update cannot precede a complete 8 s panel.

## Correctness evidence

`reports/supermimic_residual_contract_init_20261113/training_manifest.json`
records a successful Isaac initialization with:

- `obs_dim=794`, `action_dim=18`;
- actor kind `native_residual_jit_794_256_128_64`;
- zero-update action error `0.0`;
- no MPC, teacher, or BC path.

`reports/supermimic_residual_collect_smoke_20261113/` contains one native
collection step. `reports/supermimic_residual_ppo_warmup_20261113/` contains an
8-iteration, 128-environment GPU collection check (1024 native transitions).
Measured throughput was about 1.54 s per environment tick and peak allocated
GPU memory was about 2.06 GB. World-model and actor updates were correctly
gated because the warmup had not reached 102,400 native transitions.

## Current long run

The continuous run is launched as:

```text
scripts/train_supermimic_v1.py --headless \
  --output reports/supermimic_residual_ppo_train_20261113 \
  --iterations 1200 --num-envs 128 --batch-size 2048 \
  --world-updates 3 --imagine-horizon 32 --replay-capacity 250000 \
  --actor-warmup-steps 102400 \
  --init-policy-checkpoint artifacts/ppo_full_racket_residual_v4/policy.pt \
  --device cuda:0 --checkpoint-every 200 --persist-replay
```

The first 800 iterations are the complete-panel warmup. Only after that point
can the SuperMimic actor update. The run completed 1,200 iterations / 153,600
native transitions in 1,968 s. The mean native collection time was 1.56 s per
128-environment tick, the mean world-model update time was 0.035 s, the mean
imagination/BPTT time after actor updates was 0.035 s, and peak allocated GPU
memory was 4.84 GB.

The completed frozen evaluation is in
`reports/supermimic_residual_ppo_eval_20261113_retry/metrics.json`. It used the
same 128 cases, seed `882072`, GPU PhysX (`cuda:0`), 800 control ticks, and no
MPC, teacher, or BC. It completed all 128 episodes in 1,156 s with:

| metric | result |
|---|---:|
| qualified airborne string contact | 0/128 |
| ball-side complete except recovery | 0/128 |
| recovery success | 0/128 |
| complete5 | 0/128 |
| numerical failures | 0 |

This is a valid negative result for this initialization/training run. It does
not establish a task-wide upper bound: the source residual policy is marked
`formal_acceptance=false`, and its first 8-second panel did not contain a
qualified contact. A same-seed, same-protocol 128-environment evaluation of the
initial residual policy is recorded in
`reports/supermimic_residual_ppo_init_eval_20261113/metrics.json` and also
returned 0/128. The result does show that this particular SuperMimic run did
not recover the earlier 27/128 policy capability under the exact full-racket
residual execution contract; it is a zero-to-zero comparison, not evidence of
an actor improvement. The source policy is a native-executed but
`formal_acceptance=false` artifact, so a stronger accepted PPO initialization
or a corrected action-contract audit is required before interpreting later
SuperMimic updates as task progress.

The source PPO policy is not an online teacher: it is only the initialization
and frozen physical base/mixer contract. No MPC, BC, recovery-head takeover,
or hidden action override is used in the SuperMimic learner.
