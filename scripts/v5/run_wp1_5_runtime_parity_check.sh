#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--mode manual|auto|both] [--timeout-sec SEC] [--output PATH] [--bootstrap-scene] [--auto-launch-cmd CMD] [--dry-run]

Purpose:
  WP1.5 remediation B runtime parity checker for manual and auto-launch simulation startup paths.

Options:
  --mode MODE          Path mode to check: manual|auto|both.
                      Default: both
  --timeout-sec SEC    Per-topic timeout in seconds.
                      Default: 25
  --output PATH        JSON report output path.
                      Default: $REPO_ROOT/artifacts/wp1_5/runtime_parity_report.json
  --bootstrap-scene    Build ENPM662 scene packages before checks.
  --auto-launch-cmd    Explicit auto-launch command string for auto path.
  --dry-run            Print resolved command only.
  -h, --help           Show this help.
USAGE
}

MODE="both"
TIMEOUT_SEC="25"
OUTPUT="$REPO_ROOT/artifacts/wp1_5/runtime_parity_report.json"
BOOTSTRAP_SCENE=0
AUTO_LAUNCH_CMD=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --bootstrap-scene)
      BOOTSTRAP_SCENE=1
      shift
      ;;
    --auto-launch-cmd)
      AUTO_LAUNCH_CMD="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

source_if_exists() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    local old_nounset=0
    if [[ $- == *u* ]]; then
      old_nounset=1
      set +u
    fi
    # shellcheck disable=SC1090
    source "$setup_file"
    if [[ $old_nounset -eq 1 ]]; then
      set -u
    fi
  fi
}

bootstrap_scene_if_needed() {
  local scene_root="$REPO_ROOT/external/ENPM662_Group4_FinalProject"
  local scene_ws="$scene_root/src"
  local scene_setup_src="$scene_ws/install/setup.bash"
  local scene_setup_legacy="$scene_root/install/setup.bash"

  if [[ "$BOOTSTRAP_SCENE" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -d "$scene_ws" ]]; then
    echo "ERROR: scene workspace missing: $scene_ws" >&2
    return 2
  fi

  mkdir -p "$REPO_ROOT/artifacts/wp1_5"
  (
    cd "$scene_ws"
    PATH=/usr/bin:$PATH colcon build \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
      --packages-select kitchen_robot_description kitchen_robot_controller \
      --event-handlers console_direct+
  ) >"$REPO_ROOT/artifacts/wp1_5/scene_colcon_build.log" 2>&1

  source_if_exists "$scene_setup_src"
  source_if_exists "$scene_setup_legacy"
}

mkdir -p "$(dirname "$OUTPUT")"

source_if_exists /opt/ros/jazzy/setup.bash
source_if_exists "$REPO_ROOT/install/setup.bash"
source_if_exists "$REPO_ROOT/external/ENPM662_Group4_FinalProject/src/install/setup.bash"
source_if_exists "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/setup.bash"

bootstrap_scene_if_needed

CMD=(python3 -m hrl_trainer.v5.tools.runtime_parity_check --mode "$MODE" --timeout-sec "$TIMEOUT_SEC" --output "$OUTPUT")
if [[ -n "$AUTO_LAUNCH_CMD" ]]; then
  CMD+=(--auto-launch-cmd "$AUTO_LAUNCH_CMD")
fi

echo "Command: PYTHONPATH=$REPO_ROOT/hrl_ws/src/hrl_trainer ${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: runtime parity checker skipped"
  exit 0
fi

(
  cd "$REPO_ROOT"
  PYTHONPATH="hrl_ws/src/hrl_trainer:${PYTHONPATH:-}" "${CMD[@]}"
)
