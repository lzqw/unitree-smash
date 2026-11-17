#!/bin/bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=0
export QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python
bash scripts/python.sh scripts/evaluate_supermimic_v1.py --init-policy-checkpoint /home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt --init-policy-recovery-gate causal --record-physical-transitions --g07-execution-root /home/autodrive/LZQW/unitree_smash --seed 882072 --physics-device cuda:0 --device cuda:0 --headless --output /home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/run 2>&1 | tee /home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/native.log
status=${PIPESTATUS[0]}
printf "%s\n" "$status" > /home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/exit_code.txt
exit "$status"
