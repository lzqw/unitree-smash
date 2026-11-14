#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CAPTURE=/tmp/supermimic_physical_loop_capture_20261114_ub0eiq8t
PY=/home/autodrive/Isaac/env_isaaclab/bin/python
RUN="$CAPTURE/calibrated_update_16_headless_$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"
mkdir -p /home/autodrive/LZQW/video

set +e
CUDA_VISIBLE_DEVICES=0 \
QUADSMASH_PYTHON="$PY" \
bash scripts/python.sh scripts/train_supermimic_physical_block.py \
  --archive "$CAPTURE/run/physical_transitions" \
  --packed-data "$CAPTURE/packed_loop/data.pt" \
  --physical-checkpoint /tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt \
  --ball-checkpoint /tmp/supermimic_ball_sequence_20261114_axsonnit/run/ball.pt \
  --task-checkpoint "$CAPTURE/task_fit_01/task.pt" \
  --safety-mapping reports/supermimic_native_physical_loop_20261114/safety_mapping.json \
  --reference-policy-checkpoint /home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt \
  --bad-policy-checkpoint "$CAPTURE/learning_block_04/candidate_0256.pt" \
  --actor-update-budget 16 \
  --command-delta-budget 0.005 \
  --target-delta-budget-rad 0.04 \
  --video-dir /home/autodrive/LZQW/video \
  --output "$RUN" \
  --headless \
  --device cuda:0 2>&1 | tee "$RUN.log"
status=${PIPESTATUS[0]}
set -e
printf 'RUN=%s\nEXIT_CODE=%s\n' "$RUN" "$status"
exit "$status"
