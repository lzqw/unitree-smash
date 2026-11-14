# Native return calibration and constrained update

The accepted source actor is π_ref. The rejected `candidate_0256.pt` is π_bad and
is retained only for provenance and audit. The next run starts from π_ref and uses
16 accepted actor updates, the existing 0.64 s imagination window and 0.16 s
world-model window. The actor can update only the six approach tensors. A step is
rolled back together with its actor optimizer state when either the final
normalized command moves more than 0.005 from π_ref or the final joint target
moves more than 0.04 rad. These are update budgets, not actuator limits.

The command below deliberately omits `--headless`, so Isaac opens its normal
viewport. It prepares `/home/autodrive/LZQW/video` and records the request in
`video_request.json`. This branch does not claim an MP4 unless a compatible Isaac
camera/recorder extension is present; the script reports that fact instead of
creating a fake video.

```bash
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
RUN=/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/calibrated_update_16
CUDA_VISIBLE_DEVICES=0 \
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python \
bash scripts/python.sh scripts/train_supermimic_physical_block.py \
  --archive /tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/run/physical_transitions \
  --packed-data /tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/packed_loop/data.pt \
  --physical-checkpoint /tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt \
  --ball-checkpoint /tmp/supermimic_ball_sequence_20261114_axsonnit/run/ball.pt \
  --task-checkpoint /tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/task_fit_01/task.pt \
  --safety-mapping reports/supermimic_native_physical_loop_20261114/safety_mapping.json \
  --reference-policy-checkpoint /home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt \
  --bad-policy-checkpoint /tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t/learning_block_04/candidate_0256.pt \
  --actor-update-budget 16 \
  --command-delta-budget 0.005 \
  --target-delta-budget-rad 0.04 \
  --video-dir /home/autodrive/LZQW/video \
  --output "$RUN" \
  --device cuda:0
```

No `--headless` flag is intentional. The process uses the original 128 Isaac
environments and fixed 8 s task. It does not launch formal evaluation.

The first headful attempt stopped before `AFTER_APP`: Isaac Sim 5.1 crashed
in `librtx.scenedb.plugin.so` during RTX startup with
`/renderer/multiGpu/enabled=True`. No task code or checkpoint loading ran.
For the next attempt, keep the viewport enabled but add
`--rendering_mode performance` and
`--kit_args "--renderer/multiGpu/enabled=false"`. These change only the Kit
rendering startup path; PhysX, task physics, rewards, control frequency and
the learning algorithm remain unchanged.
