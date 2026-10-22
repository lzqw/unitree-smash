#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/run_physics_validation.py --output reports/physics_validation.json
"${PYTHON_BIN}" scripts/train_ppo.py --steps 8 --num-envs 32 --checkpoint checkpoints/smoke.npz --metrics reports/smoke_train.json
"${PYTHON_BIN}" scripts/evaluate_policy.py --checkpoint checkpoints/smoke.npz --episodes 4 --output reports/smoke_eval.json
echo "smoke test complete"

