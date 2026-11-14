#!/bin/bash
set -o pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/evaluate_supermimic_ball_geometry_preview.py --ball-receipt reports/supermimic_ball_modes_v2_20261114/artifact_receipt.json --output /tmp/supermimic_ball_geometry_20261114_a5tybd8n/run 2>&1 | tee /tmp/supermimic_ball_geometry_20261114_a5tybd8n/eval.log
code=${PIPESTATUS[0]}
printf "%s\n" "$code" > /tmp/supermimic_ball_geometry_20261114_a5tybd8n/exit_code.txt
exit "$code"
