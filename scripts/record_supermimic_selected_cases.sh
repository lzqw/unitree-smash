#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/unitree_smash_task_horizon_v2_20261113_174158_1440290
PY=/home/autodrive/Isaac/env_isaaclab/bin/python
CHECKPOINT=${1:?usage: $0 CHECKPOINT [--supermimic-checkpoint PATH] [--offline-render] [INDEX ...]}
shift
SUPERMIMIC_ARGS=()
OFFLINE=false
INDICES=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --supermimic-checkpoint)
      [ "$#" -ge 2 ] || { echo "--supermimic-checkpoint requires a path" >&2; exit 2; }
      SUPERMIMIC_ARGS+=(--supermimic-checkpoint "$2")
      shift 2
      ;;
    --offline-render)
      OFFLINE=true
      shift
      ;;
    *)
      INDICES+=("$1")
      shift
      ;;
  esac
done
[ "${#INDICES[@]}" -gt 0 ] || INDICES=(0 32 64 96)
STAMP=$(date +%Y%m%d_%H%M%S)
BASE=/home/autodrive/LZQW/video/selected_cases_${STAMP}_$$
mkdir -p "$BASE"

for index in "${INDICES[@]}"; do
  out="$BASE/case_${index}"
  mkdir -p "$out"
  log="$BASE/case_${index}.log"
  if [ "$OFFLINE" = true ]; then
    MODE_ARGS=(--headless --offline-render)
  else
    MODE_ARGS=(--headless --enable_cameras)
  fi
  set +e
  QUADSMASH_PYTHON="$PY" bash "$ROOT/scripts/python.sh" \
    "$ROOT/scripts/record_supermimic_selected_case.py" \
    --checkpoint "$CHECKPOINT" "${SUPERMIMIC_ARGS[@]}" \
    --output "$out" --training-index "$index" \
    --execution-root /home/autodrive/LZQW/unitree_smash \
    --fps 30 --width 640 --height 360 --rendering_mode performance \
    --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/asyncInit=false --/renderer/gpuEnumeration/glInterop/enabled=false" \
    "${MODE_ARGS[@]}" --device cuda:0 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  printf '{"training_index":%s,"exit_code":%s,"output":"%s"}\n' "$index" "$status" "$out" >> "$BASE/manifest.jsonl"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
  if [ ! -s "$out/metadata.json" ] || [ -s "$out/failure.txt" ]; then
    echo "Recorder exited without a valid capture: $out" >&2
    exit 1
  fi
  if [ "$OFFLINE" = true ]; then
    set +e
    QUADSMASH_PYTHON="$PY" bash "$ROOT/scripts/python.sh" \
      "$ROOT/scripts/render_supermimic_offline_video.py" \
      --input "$out" --output "$out/episode.mp4" --training-index "$index" \
      2>&1 | tee -a "$log"
    render_status=${PIPESTATUS[0]}
    set -e
    if [ "$render_status" -ne 0 ]; then
      exit "$render_status"
    fi
  fi
done

echo "VIDEO_ROOT=$BASE"
