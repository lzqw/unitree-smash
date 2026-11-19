# Hybrid WM contract fix and performance check

## Scope

This change only repairs the Hybrid ball transition and its validation tooling. It does not run actor, world-model, critic, or Isaac policy training updates.

The code is based on commit `e7d307be2335c5b66e8ce363450fafa80867262b` before this report is committed. The worktree contains unrelated pre-existing changes in task and evaluation files. They were not staged.

## Implemented fixes

- The endpoint relative tangent in `BallImpulsePriorV1` now uses the endpoint ball velocity. Position and surface velocity are no longer subtracted.
- Hermite sweep tangents use the configured control-period duration.
- Collision search and post-impact integration use the same drag-aware substep trajectory.
- Predicted endpoint racket pose, angular velocity, and surface velocity are used for the impact-time response.
- Hybrid legacy correction is restricted to the explicit `other_residual` mode probability. Racket, ground, and net legacy heads cannot apply a second contact impulse.
- The action-conditioned residual receives the free-flight ball endpoint. It does not re-use the already impacted analytic endpoint as its collision gate.
- Launch initialization has priority over the analytic impulse path.
- The trainable residual penalty is computed from `action_residual` alone. It does not penalize the fixed analytic impulse.
- All fitting, native-response fitting, loading, imagination, and benchmarking paths use the same registered Hybrid prior attachment helper and manifest.
- Re-attaching the same compatible frozen prior is idempotent and rejects a mismatched prior type or manifest.
- The benchmark now slices the packed archive before device transfer, measures forward and backward time at horizons 1/16/32/64, records peak CUDA allocation, and binds the output to all checkpoint hashes.

## Tests

Focused CPU tests:

```text
42 passed
```

The focused set covers Hermite time units, zero-velocity paths, launch priority, Hybrid decomposition, mode-probability gating, checkpoint loading, reset-safe negative samples, and the Hybrid prior contract.

## Offline Hybrid fit already completed

The latest offline fit is stored at:

`/home/autodrive/LZQW/experiments/supermimic_hybrid_wm_stability_20261119_e7d307b_u32`

It ran 32 residual updates with actor/world/critic updates equal to zero. No residual checkpoint was auto-promoted. The resulting zero-residual Hybrid baseline checkpoint has SHA256:

`6f781dc36028812efc51df018c3c47d9b175cacd580312a8d7fb21da842340`

Event-edge metrics for that baseline were:

| Offset | Samples | Ball velocity RMSE | Forward-sign error | Position RMSE |
|---:|---:|---:|---:|---:|
| -1 | 23 | 0.0337840 m/s | 0 | 0.000111287 m |
| 0 | 23 | 1.3759477 m/s | 0.130435 | 0.00766118 m |
| +1 | 23 | 0.00187178 m/s | 0 | 0.000010129 m |

The residual proposals were rejected for adjacent-event degradation. This means the stable result from this fit is the fixed Hybrid baseline. It is not a successful learned contact residual.

## Native response status

The existing native probe output is:

`/home/autodrive/LZQW/experiments/supermimic_native_probe_current_20261119_2fd782e_limit1c`

It contains four branches for one event-aligned specification and four independent lanes. Raw plus-minus next-state responses have norms of approximately 0.1684, 0.2373, 0.0476, and 0.0179 m/s.

Those branches span about three control ticks from probe start to the saved next state. A one-step Hybrid WM derivative is therefore not an apples-to-apples comparison. The current native direction and amplitude evidence is **not accepted** as a contact receipt. A new probe must save the exact executed action sequence and compare the same control horizon in both PhysX and WM.

## Runtime benchmark

CPU batch-2 benchmark:

`/home/autodrive/LZQW/experiments/supermimic_hybrid_ball_benchmark_20261119_cpu_e7d307b_v3/benchmark.json`

RTX 5080 batch-128 benchmark:

`/home/autodrive/LZQW/experiments/supermimic_hybrid_ball_benchmark_20261119_gpu_e7d307b/benchmark.json`

The GPU results were:

| Model | Horizon | Forward seconds | Backward seconds | Peak CUDA allocation |
|---|---:|---:|---:|---:|
| Legacy | 1 | 0.0180 | 0.0653 | 0.096 GB |
| Legacy | 16 | 0.2769 | 0.9376 | 0.238 GB |
| Legacy | 32 | 0.5528 | 1.7700 | 0.388 GB |
| Legacy | 64 | 1.1064 | 3.6269 | 0.688 GB |
| Hybrid | 1 | 0.0539 | 0.1070 | 0.237 GB |
| Hybrid | 16 | 0.8648 | 2.8759 | 3.221 GB |
| Hybrid | 32 | 1.7288 | 6.0147 | 6.391 GB |
| Hybrid | 64 | 3.4847 | 12.0805 | 12.732 GB |

The Hybrid path is approximately 2.1 times the Legacy 64-step backward time and uses about 12.7 GB at batch 128. This is within the 16 GB RTX 5080 capacity for this isolated benchmark, but leaves limited headroom for the complete actor/critic training graph. No performance shortcut was added that would change the physical transition semantics.

## Current decision

The Hybrid transition is now internally consistent and the stable offline baseline is reproducible. Native action-response direction and amplitude are still unverified because the existing probe horizon and saved action sequence do not match the one-step WM comparison. Contact-aware actor BPTT and large-scale policy training remain blocked until an exact-horizon native response receipt is generated.
