#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
export PYTHONPATH

OUT_DIR=${1:-"$REPO_ROOT/artifacts/task1_real_l3"}
EPISODES=${2:-3}
AUTO_RESET=${3:-""}
RESET_TIMEOUT=${4:-4.0}
SAFE_Z_MIN=${5:-0.20}
VERBOSE_DEBUG=${6:-""}
mkdir -p "$OUT_DIR"

ARTIFACT_JSON="$OUT_DIR/task1_training_rows_gazebo.json"
CHECKPOINT_JSON="$OUT_DIR/task1_checkpoint_gazebo.json"

AUTO_RESET_FLAG=""
if [[ -n "$AUTO_RESET" ]]; then
  case "$AUTO_RESET" in
    true|TRUE|1|yes|YES)
      AUTO_RESET_FLAG="--auto-reset"
      ;;
    false|FALSE|0|no|NO)
      AUTO_RESET_FLAG="--no-auto-reset"
      ;;
    *)
      echo "Unsupported AUTO_RESET value: $AUTO_RESET (use true|false)" >&2
      exit 2
      ;;
  esac
fi

VERBOSE_FLAG=""
if [[ -n "$VERBOSE_DEBUG" ]]; then
  case "$VERBOSE_DEBUG" in
    true|TRUE|1|yes|YES)
      VERBOSE_FLAG="--verbose-debug"
      ;;
    false|FALSE|0|no|NO)
      VERBOSE_FLAG=""
      ;;
    *)
      echo "Unsupported VERBOSE_DEBUG value: $VERBOSE_DEBUG (use true|false)" >&2
      exit 2
      ;;
  esac
fi

python3 -m hrl_trainer.v5.task1_train \
  --episodes "$EPISODES" \
  --reward-mode task1_main \
  --reset-timeout "$RESET_TIMEOUT" \
  --safe-z-min "$SAFE_Z_MIN" \
  --delta-q-limit-default 0.03 \
  --delta-q-limit-j2 0.02 \
  --saturation-threshold 0.95 \
  --epsilon-motion 0.002 \
  --stuck-window 3 \
  --reward-w-progress 1.0 \
  --reward-w-sat -0.3 \
  --reward-w-nomotion -0.8 \
  --artifact-output "$ARTIFACT_JSON" \
  --checkpoint-path "$CHECKPOINT_JSON" \
  ${AUTO_RESET_FLAG} \
  ${VERBOSE_FLAG}

echo "artifact=$ARTIFACT_JSON"
echo "checkpoint=$CHECKPOINT_JSON"
