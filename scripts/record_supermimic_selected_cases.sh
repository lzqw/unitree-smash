#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
PY=/home/autodrive/Isaac/env_isaaclab/bin/python
CHECKPOINT=${1:?usage: $0 CHECKPOINT [INDEX ...]}
shift
SUPERMIMIC_ARGS=()
INDICES=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --supermimic-checkpoint)
      if [ "$#" -lt 2 ]; then
        echo "--supermimic-checkpoint requires a path" >&2
        exit 2
      fi
      SUPERMIMIC_ARGS+=(--supermimic-checkpoint "$2")
      shift 2
      ;;
    *)
      INDICES+=("$1")
      shift
      ;;
  esac
done
if [ "${#INDICES[@]}" -eq 0 ]; then
  INDICES=(0 32 64 96)
fi
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=/home/autodrive/LZQW/video/selected_cases_${STAMP}_$$
mkdir -p "$BASE"

for index in "${INDICES[@]}"; do
  out="$BASE/case_${index}"
  mkdir -p "$out"
  log="$BASE/case_${index}.log"
  set +e
  QUADSMASH_PYTHON="$PY" \
  bash "$ROOT/scripts/python.sh" "$ROOT/scripts/record_supermimic_selected_case.py" \
    --checkpoint "$CHECKPOINT" \
    "${SUPERMIMIC_ARGS[@]}" \
    --output "$out" \
    --training-index "$index" \
    --execution-root /home/autodrive/LZQW/unitree_smash \
    --fps 30 --width 640 --height 360 \
    --rendering_mode performance \
    --kit_args="--renderer/multiGpu/enabled=false" \
    --headless --enable_cameras --device cuda:0 \
    2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  printf '{"training_index":%s,"exit_code":%s,"output":"%s"}\n' "$index" "$status" "$out" >> "$BASE/manifest.jsonl"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "VIDEO_ROOT=$BASE"
