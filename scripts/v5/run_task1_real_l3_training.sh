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
DRY_RUN_ARG=${7:-""}
mkdir -p "$OUT_DIR"

ARTIFACT_JSON="$OUT_DIR/task1_training_rows_gazebo.json"
CHECKPOINT_JSON="$OUT_DIR/task1_checkpoint_gazebo.json"

AUTO_RESET_FLAG=""
AUTO_RESET_EFFECTIVE="task1_train default"
if [[ -n "$AUTO_RESET" ]]; then
  case "$AUTO_RESET" in
    true|TRUE|1|yes|YES)
      AUTO_RESET_FLAG="--auto-reset"
      AUTO_RESET_EFFECTIVE="true (--auto-reset)"
      ;;
    false|FALSE|0|no|NO)
      AUTO_RESET_FLAG="--no-auto-reset"
      AUTO_RESET_EFFECTIVE="false (--no-auto-reset)"
      ;;
    *)
      echo "Unsupported AUTO_RESET value: $AUTO_RESET (use true|false)" >&2
      exit 2
      ;;
  esac
fi

VERBOSE_FLAG=""
VERBOSE_EFFECTIVE="false (default)"
if [[ -n "$VERBOSE_DEBUG" ]]; then
  case "$VERBOSE_DEBUG" in
    true|TRUE|1|yes|YES)
      VERBOSE_FLAG="--verbose-debug"
      VERBOSE_EFFECTIVE="true (--verbose-debug)"
      ;;
    false|FALSE|0|no|NO)
      VERBOSE_FLAG=""
      VERBOSE_EFFECTIVE="false"
      ;;
    *)
      echo "Unsupported VERBOSE_DEBUG value: $VERBOSE_DEBUG (use true|false)" >&2
      exit 2
      ;;
  esac
fi

DRY_RUN=false
if [[ "$DRY_RUN_ARG" == "--dry-run" || "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=true
fi

CMD=(
  python3 -m hrl_trainer.v5.task1_train
  --episodes "$EPISODES"
  --reward-mode task1_main
  --reset-timeout "$RESET_TIMEOUT"
  --safe-z-min "$SAFE_Z_MIN"
  --delta-q-limit-default 0.03
  --delta-q-limit-j2 0.02
  --saturation-threshold 0.95
  --epsilon-motion 0.002
  --stuck-window 3
  --reward-w-progress 1.0
  --reward-w-sat -0.3
  --reward-w-nomotion -0.8
  --artifact-output "$ARTIFACT_JSON"
  --checkpoint-path "$CHECKPOINT_JSON"
)

if [[ -n "$AUTO_RESET_FLAG" ]]; then
  CMD+=("$AUTO_RESET_FLAG")
fi
if [[ -n "$VERBOSE_FLAG" ]]; then
  CMD+=("$VERBOSE_FLAG")
fi

ROS_SETUP_STATUS="not-detected"
if [[ -n "${ROS_DISTRO:-}" || -n "${ROS_VERSION:-}" || -n "${ROS_PACKAGE_PATH:-}" || -n "${AMENT_PREFIX_PATH:-}" ]]; then
  ROS_SETUP_STATUS="detected"
fi

echo "=== task1_real_l3_training: effective configuration ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "ARTIFACT_JSON=$ARTIFACT_JSON"
echo "CHECKPOINT_JSON=$CHECKPOINT_JSON"
echo "EPISODES=$EPISODES"
echo "RESET_TIMEOUT=$RESET_TIMEOUT"
echo "SAFE_Z_MIN=$SAFE_Z_MIN"
echo "AUTO_RESET_EFFECTIVE=$AUTO_RESET_EFFECTIVE"
echo "VERBOSE_EFFECTIVE=$VERBOSE_EFFECTIVE"
echo "DRY_RUN=$DRY_RUN"
echo "PYTHONPATH=$PYTHONPATH"
echo "ROS_SETUP_STATUS=$ROS_SETUP_STATUS"
echo "ROS_DISTRO=${ROS_DISTRO:-<unset>}"

echo "=== final command (copy/paste) ==="
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] command was printed but not executed"
  exit 0
fi

"${CMD[@]}"

echo "artifact=$ARTIFACT_JSON"
echo "checkpoint=$CHECKPOINT_JSON"
