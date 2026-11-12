# SuperMimic GPU-first implementation and run status

Date: 2026-11-12 (Asia/Shanghai)
Repository: `unitree-smash`
Branch: `baseline-9cb7983-restore`
Implementation commits: `b5e302f`, `8fb0a59`, `4376588`, `65bdd94`

## Implementation

This branch adds an independent SuperMimic-style learner for the full Go2 + PiPER + racket task. It is intentionally separate from the existing PPO, MPC, teacher, BC, recovery-head, and old-policy paths.

- Random initialization: no old policy checkpoint, teacher, MPC, BC data, or recovery head is loaded.
- One 18-joint actor emits absolute joint-position targets from the ground through the complete 8-second episode.
- Native Isaac/PhysX remains the environment and low-level PD remains the actuator interface.
- Actor, critic, world model, imagination/BPTT, and replay sampling run on `cuda:0`.
- Replay storage is GPU-resident; CPU materialization is only used during low-frequency checkpoint save when replay persistence is requested.
- `--resume` restores actor, critic, world model, optimizer states, update count, and replay when the source checkpoint contains replay.
- The training loop records native collection, world-model updates, imagination/BPTT, data transfer, checkpoint/logging time, per-iteration time, and CUDA peak allocated/reserved memory.
- CUDA events are synchronized around timed sections so reported durations include asynchronous GPU work.
- Formal evaluation uses 128 environments, 800 control ticks, the unchanged 8-second physical task, and no MPC/teacher/BC intervention.

## Checkpoint and resume state

`reports/supermimic_v1_smoke11/checkpoint.pt` is a valid from-scratch SuperMimic checkpoint used as the current resume/evaluation source. It contains random-initialized model/optimizer state after one native step and has no old checkpoint provenance. Because that smoke checkpoint was created before replay persistence was enabled, its replay is empty; subsequent checkpoints created with `--persist-replay` contain the GPU replay state for a full resume.

## Current formal evaluation

Output directory: `reports/supermimic_v1_native_eval_20261112_detached3`

- Process: detached native evaluation, 128 environments, seed `882072`, 800 ticks.
- Checkpoint: `reports/supermimic_v1_smoke11/checkpoint.pt`.
- Sweep audit: 256 iterations, preserving the native contact/event path while avoiding the 4096-iteration full-racket diagnostic budget.
- Start snapshot and source hashes are saved in the output directory.
- The detached evaluator later terminated without producing `native/completed.json`, `metrics.json`, or `episodes.csv`. The wrapper recorded `status=FAILED`, `exit_code=0`, `timed_out=false`, `charged_env_steps=102400`, `charged_gpu_hours=7.6674`, and `receipt_present=false`; the orphaned child was then stopped. This is an incomplete evaluation and has no valid score.
- The earlier timed-out/interactive evaluations remain historical failures and are not combined with this run.

## GPU and process snapshot

Latest `nvidia-smi` snapshot while the evaluator was active:

- GPU: NVIDIA GeForce RTX 5080
- GPU utilization: approximately 6%
- Memory utilization: 0%
- Memory used: approximately 2.9 GB / 16.3 GB
- SuperMimic evaluator allocation: approximately 1.46 GB
- The evaluation is CPU/PhysX/event-audit bound; low GPU utilization is expected for this native evaluation path.
- Unrelated teacher-only evaluation processes were terminated to protect resources. CUDA MPS and the desktop service were left intact.
- No MPC process was detected in the final process snapshot; no MPC process was stopped.

## Next execution order

1. The detached evaluation has ended as an incomplete/failed run; preserve its logs and do not treat it as a score.
2. After GPU release, launch GPU-first checkpoint-resume training with a fresh output directory, GPU replay, CUDA timing, and periodic persisted checkpoints.
3. Use measured native/model/imagination timings and peak memory to adjust environment count, imagination batch, and replay capacity without changing physics or the formal 128-environment evaluation protocol.
4. Freeze a trained checkpoint and run one complete 128-environment native evaluation, reporting qualified contact, ball-side completion, recovery, complete5, numerical failures, and intervention counts.

No result is inferred from partial logs, training reward, solver acceptance, or an incomplete evaluation.
