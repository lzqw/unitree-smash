#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent four-cycle Recovery->Approach run with a frozen 128-env gate after
# each cycle. This script is intentionally launch-only: it is never run by the
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
mkdir -p "$RUN_ROOT" "$RUN_ROOT/tensorboard"
TB_PID=""
cleanup(){
  if [[ -n "$TB_PID" ]]; then kill "$TB_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "RUN_ROOT=$RUN_ROOT"
echo "GPU=${CUDA_VISIBLE_DEVICES:-all} DEVICE=cuda:0"
echo "PYTHON=$PYTHON"
echo "CONFIG=$ROOT/configs/supermimic_phase1_long_v1.json"

tensorboard --logdir "$RUN_ROOT/tensorboard" --host 127.0.0.1 --port "${SM_TENSORBOARD_PORT:-6006}" >"$RUN_ROOT/tensorboard.log" 2>&1 &
TB_PID=$!

PREV=""
PREV_RETURN=""
DECLINE_STREAK=0
FLAT_STREAK=0
RETURN_DROP_TOL="${SM_RETURN_DROP_TOL:-0.10}"
CONVERGENCE_TOL="${SM_RETURN_CONVERGENCE_TOL:-0.05}"
CONVERGENCE_PATIENCE="${SM_RETURN_CONVERGENCE_PATIENCE:-2}"
for cycle in 1 2 3 4; do
  OUT="$RUN_ROOT/cycle_${cycle}"
  TRAIN_ARGS=(
    --archive "$ARCHIVE" --packed-data "$PACKED"
    --physical-checkpoint "$PHYSICAL" --ball-checkpoint "$BALL"
    --task-checkpoint "$TASK" --safety-mapping "$SAFETY"
    --reference-policy-checkpoint "$REFERENCE"
    --recovery-update-budget 64 --approach-update-budget 64
    --recovery-proposal-budget 64 --approach-proposal-budget 64
    --update-mode sequential --ball-side-mode shadow
    --cycles 1 --native-ticks-per-actor-update 16
    --replay-capacity-ticks 16384 --refresh-ticks 0
    --refresh-world-updates 0 --refresh-critic-updates 0
    --command-delta-budget 0.005 --target-delta-budget-rad 0.04
    --tensorboard-dir "$RUN_ROOT/tensorboard"
    --output "$OUT" --headless --device cuda:0
    --rendering_mode performance --kit_args=--renderer/multiGpu/enabled=false
  )
  if [[ -n "$PREV" ]]; then
    TRAIN_ARGS+=(--resume-training "$PREV" --resume-warmup-ticks 0)
  fi
  echo "[cycle $cycle] training start $(date -Is)"
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QUADSMASH_PYTHON="$PYTHON" \
    bash scripts/python.sh scripts/train_supermimic_physical_block.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$OUT.log")
  ACCEPTED_COUNT="$($PYTHON - "$OUT/training_completed.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); print(int(m.get('actor_updates',0)))
PY
  )"
  if [[ "$ACCEPTED_COUNT" != 128 ]]; then
    echo "cycle $cycle accepted only $ACCEPTED_COUNT/128 proposals; preserving output and stopping" | tee "$RUN_ROOT/STOPPED"
    exit 22
  fi
  CANDIDATE="$OUT/cycle_01_candidate_$(printf '%04d' "$ACCEPTED_COUNT").pt"
  [[ -f "$CANDIDATE" ]] || { echo "missing candidate: $CANDIDATE" >&2; exit 20; }
  EVAL="$RUN_ROOT/eval_cycle_${cycle}"
  echo "[cycle $cycle] evaluation start $(date -Is)"
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" QUADSMASH_PYTHON="$PYTHON" \
    bash scripts/python.sh scripts/evaluate_supermimic_v1.py --checkpoint "$CANDIDATE" \
      --output "$EVAL" --physics-device cuda:0 --headless \
      --rendering_mode performance --kit_args=--renderer/multiGpu/enabled=false 2>&1 | tee "$EVAL.log")
  python_bin="$PYTHON"
  ACCEPTED="$($python_bin - "$EVAL/metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); n=m.get('native_episode_metrics',m)
ok=(float(n.get('native_episode_return_mean',-1e99))>-12.52678435 and
    int(n.get('ball_side_complete_except_recovery',-1))>=53 and
    int(n.get('complete5',-1))>=27 and int(m.get('numerical_failures',-1))==0)
print('1' if ok else '0')
PY
  )"
  echo "[cycle $cycle] accepted=$ACCEPTED"
  if [[ "$ACCEPTED" != 1 ]]; then
    echo "cycle $cycle rejected; preserving output and stopping" | tee "$RUN_ROOT/STOPPED"
    exit 21
  fi
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
done
echo "PHASE1_FINISHED=$RUN_ROOT"
