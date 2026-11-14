#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
PY=/home/autodrive/Isaac/env_isaaclab/bin/python
CHECKPOINT=${1:?usage: $0 CHECKPOINT [INDEX ...]}
shift
if [ "$#" -eq 0 ]; then
  INDICES=(0 32 64 96)
else
  INDICES=("$@")
fi
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=/home/autodrive/LZQW/video/selected_cases_${STAMP}
mkdir -p "$BASE"

for index in "${INDICES[@]}"; do
  out="$BASE/case_${index}"
  mkdir -p "$out"
  set +e
  CUDA_VISIBLE_DEVICES=0 \
  QUADSMASH_PYTHON="$PY" \
  bash "$ROOT/scripts/python.sh" "$ROOT/scripts/record_supermimic_selected_case.py" \
    --checkpoint "$CHECKPOINT" \
    --output "$out" \
    --training-index "$index" \
    --execution-root /home/autodrive/LZQW/unitree_smash \
    --fps 30 --width 640 --height 360 \
    --headless --enable_cameras --device cuda:0 \
    2>&1 | tee "$out/run.log"
  status=${PIPESTATUS[0]}
  set -e
  printf '{"training_index":%s,"exit_code":%s,"output":"%s"}\n' "$index" "$status" "$out" >> "$BASE/manifest.jsonl"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "VIDEO_ROOT=$BASE"
