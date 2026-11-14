#!/bin/bash
set -o pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/fit_supermimic_ball_sequence.py --source-receipt reports/supermimic_ball_geometry_preview_20261114/artifact_receipt.json --output /tmp/supermimic_ball_sequence_20261114_axsonnit/run --updates 200 --batch 16 2>&1 | tee /tmp/supermimic_ball_sequence_20261114_axsonnit/fit.log
code=${PIPESTATUS[0]}
printf "%s\n" "$code" > /tmp/supermimic_ball_sequence_20261114_axsonnit/exit_code.txt
exit "$code"
