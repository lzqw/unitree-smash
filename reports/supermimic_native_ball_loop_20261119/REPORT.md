# Native ball loop status

## Scope

This report records the fixed-seed native probe, offline ball-response fits, and hybrid world-model validation completed on 2026-11-19. No Actor, Critic, or online World Model training was started.

## Engineering fixes

- `run_supermimic_event_aligned_native_probes.py` now accepts `--seed` and forwards the exact seed to every Isaac child. The probe archive and child runs therefore share seed `882073`.
- `fit_supermimic_native_probe_response.py` now supports a positive direction-loss weight and selects eligible holdout iterates by response direction cosine before RMSE. The selection rule is stored in the candidate metadata.
- Unrelated dirty files in the worktree were preserved.

## Native probe evidence

- Baseline archive: `/home/autodrive/LZQW/experiments/supermimic_native_probe_validation_20261119_seed882073`
- Valid probe output: `gpu_probe_seed882073_limit1_seedfixed`
- Probe branches: `nominal_a`, `nominal_b`, `minus`, `plus`
- All four branches used seed `882073`, GPU PhysX `cuda:0`, and the same event spec.
- Contact bundle: `bundles_seedfixed/contact3_seed882073_t267.pt`
- Contact bundle SHA256: `f57eb052de0e63ebd6b1ebe9b3f04c3154bf88f33a329cad954bdee3ed6eb586`
- This is paired native evidence for three supported lanes. It is not yet a passing receipt because the response-direction acceptance bounds and independent event coverage are not satisfied.

## Offline native-response fit

The direction-first candidate was fitted from 12 training response samples and evaluated on 4 supported holdout samples.

- Candidate output: `temporal_response_fit_v11_seedfixed_train12_val4supported_lr5e4_dir2_select`
- Response RMSE: `0.0306099057`
- Zero-response baseline RMSE: `0.0337354988`
- Minimum response cosine: `0.5497263`
- Maximum relative error: `0.9283188`
- Actor updates: `0`
- World updates: `0`
- Critic updates: `0`

The candidate is still marked as an offline fit. The native contact receipt remains unissued until event alignment and native Jacobian acceptance are independently checked against declared bounds.

## Hybrid action-conditioned ball fit

- Candidate: `action_ball_fit_contact_v3_b64_u32/action_ball_candidate.pt`
- Candidate SHA256: `449d1cd3c9ad0d8b0d4395f70ea0492d8d5181965c9022cb55e1aa2b0e0d0c2c`
- GPU fit updates: `32`
- Actor, World Model, and Critic updates: `0`
- GPU peak memory: `873065984` bytes
- Elapsed time: `289.52` seconds
- Training contact transitions: `89`
- Hard precontact transitions: `89`

On the held-out contact-neighborhood split, the candidate changed the event-aligned velocity metrics as follows.

| Relative event tick | Initial velocity RMSE | Candidate velocity RMSE | Initial forward-sign error | Candidate forward-sign error |
|---:|---:|---:|---:|---:|
| -1 | 4.0122 | 4.1682 | 0.3913 | 0.3913 |
| 0 | 4.1869 | 3.3935 | 0.7391 | 0.4783 |
| +1 | 0.04360 | 0.04361 | 0 | 0 |

The contact-tick prediction improved, but the precontact error did not. The candidate therefore remains offline-only and does not unlock contact BPTT.

## Hybrid WM validation

The validation command completed on GPU without Isaac simulation. The event-aligned held-out windows still show substantial multi-step error and the validation tool reports `native_simulation=false`. The command-sensitivity report for the legacy path has zero direct ball derivative. The offline candidate's nonzero differentiable response is not evidence of correct PhysX direction.

## Tests

```text
22 passed in 0.90s
```

The changed scripts also pass `py_compile` with the Isaac Python environment.

## Decision

The next phase after all gates pass is a bounded native canary. It will freeze the validated hybrid WM, run one Recovery block, refresh WM and same-policy Critic labels from complete native episodes, run one Approach block, refresh again, and then run a second Recovery block. Each block will record actual Actor parameter changes, final-command changes, WM source/stage usage, and native return. Only after this R8→A8→R8 canary passes will the 256-update large training budget be enabled.

Current state remains:

```text
offline_ball_fit_complete
event_aligned_improvement_partial
native_jacobian_receipt_pending
contact_bptt_blocked
large_actor_training_not_started
```
