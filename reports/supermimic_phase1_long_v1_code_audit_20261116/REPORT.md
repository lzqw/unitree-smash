# SuperMimic phase-1 long-run code audit

Date: 2026-11-16 (Asia/Shanghai)

## Source and scope

- Worktree: `/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290`
- Base before this change: `d37254c5bb068f576ed4022ab1228d70c4f8eb30`
- Verified initial policy SHA256: `8f510fe9f218beef791ac21675a19ed25d11f510a83fe247cddf2893bee08399`
- No Isaac training or 128-environment evaluation was started by this audit.
- Existing untracked assets and historical reports were left untouched.

## Changes

1. `PhysicalLoopReplay` is now a chronological CPU ring with a 16384-tick default, reset/stream-safe windows, stage start buckets (prepare, approach, contact, touchdown, recovery, terminal), and a bounded recent-event archive.
2. The training entry point supports persistent TensorBoard logging, four-cycle sequential scheduling, configurable native ticks per actor update, replay capacity, and resume without an implicit episode-length warmup.
3. Recovery and Approach have independent Adam state dictionaries; checkpoints include global proposal counters, replay position metadata, and both head optimizers.
4. Fresh native collection is placed in a new replay stream so fixed-panel episode-index reuse cannot create cross-reset windows.
5. `run_supermimic_phase1_long_v1.sh` is a launch-only orchestrator. It runs four 64-Recovery/64-Approach cycles, performs one frozen 128-environment gate after each cycle, and stops on the first rejected cycle. It is not invoked during validation.

## Fixed first-phase scale

- GPU: `cuda:0` (RTX 5080 expected)
- Native environments: 128
- Cycles: 4
- Per cycle: up to 64 Recovery and 64 Approach proposals/accepted updates; the proposal limit is 64 per block, so a rejection is recorded and the cycle stops short rather than silently extending the native-tick budget.
- Actor updates: 512 total (256 Recovery, 256 Approach)
- Native control ticks: 8192 total (16 per actor update)
- Native transitions: 1,048,576 (8192 x 128)
- Imagination window: 0.64 s
- World-model window: 0.16 s
- Replay: 16,384 ticks, CPU ring with GPU minibatches
- Actor/world/critic learning rates: 5e-5 / 2e-4 / 2e-4
- Formal gate: one frozen 128-env, 800-tick evaluation after each accepted cycle; no MPC, teacher, reward, asset, physics, action-map, or control-period changes.

## Validation

Interpreter: `/home/autodrive/Isaac/env_isaaclab/bin/python` through `scripts/python.sh`.

Passed:

```text
18 passed in 1.27s
```

Command:

```bash
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh -m pytest \
  tests/test_supermimic_physical_loop_components.py \
  tests/test_supermimic_physical_dataset.py \
  tests/test_supermimic_sequential_audit.py \
  tests/test_supermimic_native_sequence.py \
  tests/test_supermimic_native_metrics.py -q -ra --maxfail=1
```

Static checks passed: Python compilation, shell syntax, `git diff --check`, and a replay window smoke (`16 x 8` CPU window from the verified packed archive). The long-run orchestrator was not executed.

## Current status

Code is ready for a user-approved GPU run. No new performance result exists for this revision. The last completed R8/A8 diagnostic remains return `-12.12484468`, qualified contact `73/128`, ball-side `53/128`, recovery `38/73`, complete5 `25/128`; it was rejected because complete5 was below the accepted 27/128 gate.
