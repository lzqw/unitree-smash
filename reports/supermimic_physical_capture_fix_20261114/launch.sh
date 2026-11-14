#!/usr/bin/env bash
set -o pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
RUN_CAPTURE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/run_supermimic_evaluation.py --init-policy-checkpoint /home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt --g07-execution-root /home/autodrive/LZQW/unitree_smash --output "$RUN_CAPTURE_DIR/run" --seed 882072 --headless --device cuda:0 --physics-device cuda:0 --record-physical-transitions 2>&1 | tee "$RUN_CAPTURE_DIR/native.log"
RUN_CAPTURE_STATUS=${PIPESTATUS[0]}
printf '%s\n' "$RUN_CAPTURE_STATUS" > "$RUN_CAPTURE_DIR/exit_code.txt"
exit "$RUN_CAPTURE_STATUS"
