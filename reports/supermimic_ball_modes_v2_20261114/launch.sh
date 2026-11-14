#!/bin/bash
set -o pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/fit_supermimic_ball_residual_v2.py --source-receipt reports/supermimic_physical_contact_v2_20261114/artifact_receipt.json --trace-receipt reports/supermimic_ball_transition_audit_20261114/artifact_receipt.json --output /tmp/supermimic_ball_modes_20261114_p0m0ulo9/run --updates 600 2>&1 | tee /tmp/supermimic_ball_modes_20261114_p0m0ulo9/fit.log
code=${PIPESTATUS[0]}
printf "%s\n" "$code" > /tmp/supermimic_ball_modes_20261114_p0m0ulo9/exit_code.txt
exit "$code"
