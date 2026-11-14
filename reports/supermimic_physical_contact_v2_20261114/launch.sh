#!/usr/bin/env bash
set -uo pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/fit_supermimic_physical_world_v2.py --source-receipt reports/supermimic_physical_world_fit_20261114/artifact_receipt.json --output "$(dirname "$0")/run" --updates 200 --batch 128 2>&1 | tee "$(dirname "$0")/fit.log"
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > "$(dirname "$0")/exit_code.txt"
exit "$result"
