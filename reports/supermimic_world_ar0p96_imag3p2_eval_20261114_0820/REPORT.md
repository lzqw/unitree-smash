# Formal evaluation after autoregressive SuperMimic updates

## Protocol

- Checkpoint: `supermimic_world_ar0p96_imag3p2_20261114_0750/checkpoint.pt`
- Checkpoint SHA256: `bc6aa4334f96cd1a2ced4b134e5d1c6bc66dbb171e07d37019820186c13dd606`
- Initialization source: `/home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline/artifacts/ppo_full_racket_residual_v4/policy.pt`
- 128 fixed environments, seed `882072`, 800 control ticks / 8 seconds
- GPU PhysX: `cuda:0`, `gpu_dynamics=true`
- No MPC or teacher interventions

## Result

| Metric | Result |
|---|---:|
| Qualified airborne string contact | **20/128 (15.63%)** |
| Ball-side complete except recovery | **5/128 (3.91%)** |
| Recovery-success records | 64 (the evaluator's recovery field is not the complete-task denominator) |
| Recovery given qualified contact | 0.60 (as emitted by evaluator) |
| complete5 | **4/128 (3.13%)** |
| Numerical failures | **36,033** |
| MPC / teacher interventions | 0 / 0 |
| Elapsed | 1,204.6 s |

## Interpretation

This is a completed formal evaluation, but it is a rejected candidate. The actor had 68 imagination updates and a numerically healthy critic, yet the frozen native result is materially below the initialized policy/baseline (roughly 27/128 complete5 in the established protocol). The qualified-contact count also fell to 20/128. The update therefore did not learn a useful task policy; it damaged the initialized action contract despite the world-model and critic gates.

The high `numerical_failures` count and the unusual recovery count should be audited separately before using this evaluator output for fine-grained recovery claims. The complete5 and qualified-contact regressions are already sufficient to reject this checkpoint as a new best.

Next training should start again from the original PPO checkpoint, use a much smaller actor step and a materially stronger action anchor or an explicit frozen-policy acceptance test. This checkpoint is retained for diagnosis and is not used as the next initialization.
