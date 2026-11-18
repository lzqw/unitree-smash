# Small GPU validation readiness

## Scope

This revision only prepares and audits the event-aligned native probe path. It does not start actor training, WM fitting, or formal 128-environment evaluation.

## Code changes

- Physical-loop loading accepts an explicit `--g07-execution-root` and records the resolved root in the model manifest. Missing provenance now fails clearly instead of silently loading a different worktree.
- Training and recorded WM validation pass the explicit execution root through to the G07 controller.
- The long-run launcher requires `SM_G07_EXECUTION_ROOT` and forwards it.
- Default per-cycle actor budgets are Recovery 256 and Approach 256, with proposal budgets 512 each. These defaults do not start a run.
- Added `scripts/run_supermimic_event_aligned_native_probes.py`. It checks CUDA, runs every declared spec through nominal-a, nominal-b, minus, and plus in fresh Isaac processes, and writes `COLLECTION_FINISHED` only when all four branches produce immutable native probe records.

## CPU verification

`pytest` targeted contract set: 63 passed, 0 failed.

Python syntax checks and shell syntax checks passed.

## GPU/Isaac status

The small GPU validation was not started because the host currently reports no usable CUDA device. See `gpu_preflight.txt`. The observed state was `nvidia-smi` unable to communicate with the driver and `torch.cuda.is_available() == False` with zero devices.

No native event-aligned probe, native paired action response, or R→A→R update was claimed or produced in this run.

## Launch gate

Large-scale training remains **NO-GO** until the driver is restored, the event-aligned native probe collection reaches independent lane/episode support, the WM/native response receipt is produced, and a small R→A→R GPU canary completes. A large run must still be started only after explicit user confirmation.
