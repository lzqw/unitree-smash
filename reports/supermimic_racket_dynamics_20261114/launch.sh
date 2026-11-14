#!/usr/bin/env bash
# Fit runtime a6d3ed1; audit runtime c0ece2d. Output directories are immutable.
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/fit_supermimic_racket_dynamics.py --source-run /tmp/supermimic_ball_sequence_20261114_axsonnit/run --source-receipt reports/supermimic_ball_sequence_fit_20261114/artifact_receipt.json --output /tmp/supermimic_racket_dynamics_20261114_YRFhKS/run --updates 200 --batch 16
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/audit_supermimic_racket_dynamics_fit.py /tmp/supermimic_racket_dynamics_20261114_YRFhKS/run --output /tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/full_state_audit.json
