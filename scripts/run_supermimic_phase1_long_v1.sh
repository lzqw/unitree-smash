#!/usr/bin/env bash
set -Eeuo pipefail

# Short alternating (R8 -> A8 -> combined frozen refresh) x16, then
# one formal 128-env evaluation. Large-profile evidence gates remain mandatory. Budgets remain explicit overrides. This script is intentionally launch-only: it is never run by the
# code validation step. Set the six SM_* paths before invoking it.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QUADSMASH_PYTHON:-/home/autodrive/Isaac/env_isaaclab/bin/python}"
RUN_ROOT="${SM_RUN_ROOT:-/home/autodrive/LZQW/supermimic_phase1_$(date +%Y%m%d_%H%M%S)}"
ARCHIVE="${SM_ARCHIVE:?set SM_ARCHIVE to native physical transition archive}"
PACKED="${SM_PACKED_DATA:?set SM_PACKED_DATA to packed_loop/data.pt}"
PHYSICAL="${SM_PHYSICAL_CHECKPOINT:?set SM_PHYSICAL_CHECKPOINT to world.pt}"
BALL="${SM_BALL_CHECKPOINT:?set SM_BALL_CHECKPOINT to ball.pt}"
TASK="${SM_TASK_CHECKPOINT:?set SM_TASK_CHECKPOINT to task.pt}"
SAFETY="${SM_SAFETY_MAPPING:-$ROOT/reports/supermimic_native_physical_loop_20261114/safety_mapping.json}"
REFERENCE="${SM_REFERENCE_POLICY:-/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/frozen_v2_smooth_approach16_env5120_01/policy.pt}"
G07_ROOT="${SM_G07_EXECUTION_ROOT:?set SM_G07_EXECUTION_ROOT to the exact measured G07 execution root}"
mkdir -p "$(dirname "$RUN_ROOT")"
if ! mkdir "$RUN_ROOT"; then
  echo "Refusing to reuse training output: $RUN_ROOT; set a new SM_RUN_ROOT, including for resume." >&2
  exit 15
fi
mkdir "$RUN_ROOT/tensorboard"
REF_BANK="${SM_TASK_REFERENCE_BANK:?set SM_TASK_REFERENCE_BANK to a terminal-verified v2 bank}"
TRACK_WEIGHT="${SM_TRACKING_WEIGHT:?declare SM_TRACKING_WEIGHT before this experiment}"
R_UPDATES="${SM_RECOVERY_UPDATES:-128}"
A_UPDATES="${SM_APPROACH_UPDATES:-128}"
R_PROPOSALS="${SM_RECOVERY_PROPOSALS:-256}"
A_PROPOSALS="${SM_APPROACH_PROPOSALS:-256}"
CYCLES="${SM_CYCLES:-1}"
TB_PID=""
cleanup(){
  if [[ -n "$TB_PID" ]]; then kill "$TB_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "RUN_ROOT=$RUN_ROOT"
echo "GPU=${CUDA_VISIBLE_DEVICES:-all} DEVICE=cuda:0"
echo "PYTHON=$PYTHON"
echo "CONFIG=$ROOT/configs/supermimic_success_guided_v2.json"

"$PYTHON" -m tensorboard.main --logdir "$RUN_ROOT/tensorboard" --host 127.0.0.1 --port "${SM_TENSORBOARD_PORT:-6006}" >"$RUN_ROOT/tensorboard.log" 2>&1 &
TB_PID=$!
"$PYTHON" - "$TB_PID" "${SM_TENSORBOARD_PORT:-6006}" "$RUN_ROOT/tensorboard" <<'PYTB'
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

pid, port, logdir = sys.argv[1:]
for _ in range(60):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        sys.exit('TensorBoard exited during startup. See tensorboard.log.')
    try:
        with urlopen(f'http://127.0.0.1:{port}/data/logdir', timeout=1) as response:
            served = json.load(response)['logdir']
    except (OSError, ValueError, KeyError):
        time.sleep(1)
        continue
    if Path(served).resolve() != Path(logdir).resolve():
        sys.exit('TensorBoard port belongs to a different log directory.')
    break
else:
    sys.exit('TensorBoard did not serve this run within 60 seconds.')
PYTB

PREV=""
if [[ -n "${SM_RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$SM_RESUME_CHECKPOINT" ]] || { echo "resume checkpoint not found: $SM_RESUME_CHECKPOINT" >&2; exit 12; }
  PREV="$SM_RESUME_CHECKPOINT"
fi
PREV_EVAL="${SM_NATIVE_ACCEPTANCE_RECEIPT:-}"
PREV_RETURN=""
DECLINE_STREAK=0
FLAT_STREAK=0
RETURN_DROP_TOL="${SM_RETURN_DROP_TOL:-0.10}"
CONVERGENCE_TOL="${SM_RETURN_CONVERGENCE_TOL:-0.05}"
CONVERGENCE_PATIENCE="${SM_RETURN_CONVERGENCE_PATIENCE:-2}"
for ((cycle=1; cycle<=CYCLES; cycle++)); do
  OUT="$RUN_ROOT/cycle_${cycle}"
  TRAIN_ARGS=(
    --archive "$ARCHIVE" --packed-data "$PACKED"
    --physical-checkpoint "$PHYSICAL" --ball-checkpoint "$BALL"
    --task-checkpoint "$TASK" --safety-mapping "$SAFETY"
    --reference-policy-checkpoint "$REFERENCE" --g07-execution-root "$G07_ROOT"
    --recovery-update-budget "$R_UPDATES" --approach-update-budget "$A_UPDATES"
    --recovery-proposal-budget "$R_PROPOSALS" --approach-proposal-budget "$A_PROPOSALS"
    --update-mode sequential --sequential-layout short-alternating --head-block-size "${SM_HEAD_BLOCK_SIZE:-8}"
    --refresh-mode "${SM_REFRESH_MODE:-pair}" --ball-side-mode shadow
    --cycles 1 --native-ticks-per-actor-update 16
    --wm-bootstrap-max-ticks "${SM_WM_BOOTSTRAP_MAX_TICKS:-0}"
    --replay-capacity-ticks 16384 --refresh-ticks 800
    --refresh-world-updates 32 --refresh-critic-updates 20
    --wm-natural-flight-batch "${SM_WM_NATURAL_FLIGHT_BATCH:-0}"
    --world-updates-per-actor-update 4 --wm-batch-size "${SM_WM_BATCH_SIZE:-512}"
    --wm-stage-quota uniform=32,pre_touchdown=24,touchdown=24,post_touchdown=24,recovery_failure=16,stable_recovery=8
    --wm-approach-stage-quota uniform=24,approach_active=32,hit_prepare=24,contact_release=24,approach_failure=16,post_hit=8
    --task-reference-bank "$REF_BANK" --tracking-weight "$TRACK_WEIGHT"
    --command-delta-budget 0.005 --target-delta-budget-rad 0.04
    --tensorboard-dir "$RUN_ROOT/tensorboard"
    --native-return-drop-tolerance "$RETURN_DROP_TOL" --native-return-flat-tolerance "$CONVERGENCE_TOL"
    --native-return-patience "$CONVERGENCE_PATIENCE"
    --output "$OUT" --headless --device cuda:0
    --rendering_mode performance --kit_args=--renderer/multiGpu/enabled=false
  )
  if [[ -n "${SM_WORLD_LEARNING_RATE:-}" ]]; then
    TRAIN_ARGS+=(--world-learning-rate "$SM_WORLD_LEARNING_RATE")
  fi
  if [[ "${SM_WM_SEPARATE_SUPERVISION:-0}" == 1 ]]; then
    TRAIN_ARGS+=(--wm-separate-supervision)
  fi
  if [[ -n "${SM_WM_FORWARD_GUARD:-}" ]]; then
    TRAIN_ARGS+=(--wm-forward-guard "$SM_WM_FORWARD_GUARD")
  fi
  if [[ -n "${SM_ACTION_BALL_CHECKPOINT:-}" ]]; then
    [[ -f "$SM_ACTION_BALL_CHECKPOINT" ]] || { echo "Missing opt-in action-ball candidate: $SM_ACTION_BALL_CHECKPOINT" >&2; exit 13; }
    TRAIN_ARGS+=(--action-ball-checkpoint "$SM_ACTION_BALL_CHECKPOINT")
  fi
  if [[ -n "${SM_WM_ACTION_COLLECTION:-}" ]]; then
    # One complete collection per line; preserve spaces in paths.
    while IFS= read -r collection; do
      if [[ -n "$collection" ]]; then TRAIN_ARGS+=(--wm-action-collection "$collection"); fi
    done <<< "$SM_WM_ACTION_COLLECTION"
  fi
  if [[ "${SM_TRAINING_PROFILE:-large}" == large ]]; then
    TRAIN_ARGS+=(--training-profile large --wm-sampling-mode promotion --contact-validation-receipt "${SM_CONTACT_VALIDATION_RECEIPT:?contact validation evidence required}")
  fi
  if [[ -n "$PREV" ]]; then
    TRAIN_ARGS+=(--resume-training "$PREV" --resume-warmup-ticks 0)
    if [[ -n "$PREV_EVAL" ]]; then
      TRAIN_ARGS+=(--resume-next-cycle --native-acceptance-receipt "$PREV_EVAL")
    fi
  fi
  echo "[cycle $cycle] training start $(date -Is)"
  (cd "$ROOT" && QUADSMASH_ZSTD_ARCHIVE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QUADSMASH_PYTHON="$PYTHON" \
    bash scripts/python.sh scripts/train_supermimic_physical_block.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$OUT.log")
  ACCEPTED_COUNT="$($PYTHON - "$OUT/training_completed.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); print(int(m.get('actor_updates',0)))
PY
  )"
  if (( ACCEPTED_COUNT != R_UPDATES + A_UPDATES )); then
    echo "cycle $cycle accepted only $ACCEPTED_COUNT/$((R_UPDATES+A_UPDATES)) effective updates; preserving output and stopping" | tee "$RUN_ROOT/STOPPED"
    exit 22
  fi
  CANDIDATE="$OUT/cycle_01_candidate_$(printf '%04d' "$ACCEPTED_COUNT").pt"
  [[ -f "$CANDIDATE" ]] || { echo "missing candidate: $CANDIDATE" >&2; exit 20; }
  EVAL="$RUN_ROOT/eval_cycle_${cycle}"
  echo "[cycle $cycle] evaluation start $(date -Is)"
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QUADSMASH_PYTHON="$PYTHON" \
    bash scripts/python.sh scripts/evaluate_supermimic_v1.py --checkpoint "$CANDIDATE" \
      --output "$EVAL" --g07-execution-root "$G07_ROOT" --physics-device cuda:0 --headless \
      --rendering_mode performance --kit_args=--renderer/multiGpu/enabled=false 2>&1 | tee "$EVAL.log")
  python_bin="$PYTHON"
  if ! (cd "$ROOT" && QUADSMASH_PYTHON="$PYTHON" bash scripts/python.sh scripts/verify_supermimic_promotion.py \
      --metrics "$EVAL/metrics.json" --checkpoint "$CANDIDATE" --output "$EVAL/promotion.json"); then
    echo "cycle $cycle promotion rejected; preserving evidence and stopping" | tee "$RUN_ROOT/STOPPED"
    exit 21
  fi
  echo "[cycle $cycle] accepted=1 (checkpoint, execution contract and complete protocol verified)"
  CURRENT_RETURN="$($python_bin - "$EVAL/metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); n=m.get('native_episode_metrics',m)
print(float(n['native_episode_return_mean']))
PY
  )"
  if [[ -n "$PREV_RETURN" ]]; then
    read -r DECLINE_STREAK FLAT_STREAK <<EOF
$($python_bin - "$PREV_RETURN" "$CURRENT_RETURN" "$DECLINE_STREAK" "$FLAT_STREAK" "$RETURN_DROP_TOL" "$CONVERGENCE_TOL" <<'PY'
import sys
prev,cur,decline,flat,drop,flat_tol=map(float,sys.argv[1:])
decline=int(decline)+1 if cur < prev-drop else 0
flat=int(flat)+1 if abs(cur-prev) <= flat_tol else 0
print(decline,flat)
PY
)
EOF
    echo "[cycle $cycle] native_return=$CURRENT_RETURN previous=$PREV_RETURN decline_streak=$DECLINE_STREAK flat_streak=$FLAT_STREAK"
    if (( DECLINE_STREAK >= 2 )); then
      echo "native return declined for two consecutive cycle evaluations; stopping" | tee "$RUN_ROOT/STOPPED_RETURN_DECLINE"
      exit 23
    fi
    if (( FLAT_STREAK >= CONVERGENCE_PATIENCE )); then
      echo "native return converged within tolerance; stopping" | tee "$RUN_ROOT/STOPPED_CONVERGED"
      exit 24
    fi
  fi
  PREV_RETURN="$CURRENT_RETURN"
  PREV="$CANDIDATE"
  PREV_EVAL="$EVAL/metrics.json"
done
echo "PHASE1_FINISHED=$RUN_ROOT"
