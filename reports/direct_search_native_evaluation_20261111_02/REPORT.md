# Direct-search native evaluation (entry-fixed attempt)

Date: 2026-11-11 (Asia/Shanghai)

## Result

`NATIVE_EVALUATION_ABORTED_BEFORE_RESET`.

This was the only native evaluation allowed for this round. The new output directory is independent of the previous failed run and the previous directory was not modified. Isaac initialized the 128-environment scene, and the formal entry check passed before any environment reset or step:

- root: `/home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline`
- module: `scripts.native_pure_control_state_driven_long_mpc_v1_qp_continuity_direct_search`
- adapter file: `/home/autodrive/LZQW/unitree_smash_contact_recovery_9cb_baseline/scripts/native_pure_control_state_driven_long_mpc_v1_qp_continuity_direct_search.py`
- class: `NativePureControlStateDrivenLongMPCV1QPContinuityDirectSearch`
- commit: `482f7b314513dfb6f2fd1d53b3fd8e81d33959d4`
- old repair module used for adapter: `false`

The process then failed during the existing asset-manifest admission check:

```text
ValueError: USD dependency missing or changed:
assets/robots/piper_standard/configuration/piper_standard_base.usd
```

The current worktree contains the tracked manifest/configuration but not the ignored USD payload. No fallback to another worktree was used, because that would violate the entry provenance requirement. Exit code was 0 at the wrapper level, but `native/completed.json` was absent. Last controller tick: not started. Completed cases: 0.

No second run, seed, parameter scan, or algorithm change was started after this failure.

## CPU evidence retained

The direct-search CPU gate remains valid: 7/74 complete 32-node candidates were formed (TRAIN 7, 17, 23, 32, 37, 60, 68), with branch-local final-command history and original limiter preview. These are diagnostic candidates only and are not native successes.

## Metrics

The candidate has no native metrics. The authoritative saved comparisons remain:

| Metric | Historical 9cb7983 | Work comparator 7251b5f | Direct candidate |
|---|---:|---:|---:|
| Conditional recovery | 52/74 (70.27%) | 55/74 (74.32%) | not evaluated |
| Complete5 | 40/128 (31.25%) | 39/128 (30.47%) | not evaluated |
| Qualified aerial hit | saved baseline value | 74/128 | not evaluated |
| Ball-side conditions except recovery | saved baseline value | 46/128 | not evaluated |
| Post-takeover head-free and recovered | 0/74 | 0/74 | not evaluated |
| Post-takeover non-foot-free and recovered | 0/74 | 0/74 | not evaluated |
| Numerical faults / command audit | saved audit passed | 0; audit passed | not evaluated |

The historical complete5 value is kept as 40/128 from the original metrics; it is not replaced by the later 39/128 work-comparator value.
