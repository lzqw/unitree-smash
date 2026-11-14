#!/usr/bin/env bash
# Exact commands used from the debug worktree, original unique output retained.
# First command at commit 0a206f3; diagnostic at commit 759c40a.
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/audit_supermimic_ball_impulse_prior.py --source-run /tmp/supermimic_ball_sequence_20261114_axsonnit/run --output /tmp/supermimic_ball_impulse_20261114_rRn8Sy/run
QUADSMASH_PYTHON=/home/autodrive/Isaac/env_isaaclab/bin/python bash scripts/python.sh scripts/audit_supermimic_impulse_robot_oracle.py /tmp/supermimic_ball_impulse_20261114_rRn8Sy/run --output /tmp/supermimic_ball_impulse_20261114_rRn8Sy/run/robot_oracle.json
