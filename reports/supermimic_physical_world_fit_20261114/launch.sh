#!/usr/bin/env bash
set -uo pipefail
cd /tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
CUDA_VISIBLE_DEVICES=0 QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/fit_supermimic_physical_world.py /tmp/supermimic_physical_capture_fix_20261114_Rtntwh/run/physical_transitions --output "$(dirname "$0")/run" --updates 200 --batch 128 --seed 971 --device cuda:0 2>&1 | tee "$(dirname "$0")/fit.log"
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > "$(dirname "$0")/exit_code.txt"
exit "$result"
