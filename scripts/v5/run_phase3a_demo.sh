#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQUEST_JSON="$REPO_ROOT/artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json"
OUTPUT_JSON="$REPO_ROOT/artifacts/v5/phase3a_demo/phase3a_runtime_plan.json"
SCENE_LOG="$REPO_ROOT/artifacts/v5/phase3a_demo/kitchen_scene_launch.log"
MODE="dry_run"
LAUNCH_SCENE=0
SKIP_BRIDGE_VALIDATION=0
DRY_RUN=0
SCENE_MODE="headless"
STARTUP_WAIT_SEC=12
COMMAND="Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose."

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--launch-scene] [--scene-mode headless|gui] [--mode dry_run|ros_dry_run] [--dry-run]

Purpose:
  Run the Phase 3A Qwen L1 -> Approach/Finisher runtime demo path using the
  existing V5 Gazebo bridge/launch scripts.

Options:
  --launch-scene         Start kitchen scene through scripts/v5/launch_kitchen_scene.sh.
  --skip-bridge-validation
                         Skip external kitchen scene symlink validation.
  --scene-mode MODE      headless (default) or gui.
  --mode MODE            dry_run (default) or ros_dry_run.
  --request-json PATH    Existing Qwen L1 request artifact.
  --output-json PATH     Runtime plan output JSON.
  --scene-log PATH       Kitchen scene launch log when --launch-scene is used.
  --startup-wait SEC     Seconds to wait after launching scene. Default: $STARTUP_WAIT_SEC.
  --command TEXT         Demo command if request artifact must be regenerated.
  --dry-run              Print commands and skip long-running launch/runtime calls.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch-scene)
      LAUNCH_SCENE=1; shift ;;
    --skip-bridge-validation)
      SKIP_BRIDGE_VALIDATION=1; shift ;;
    --scene-mode)
      SCENE_MODE="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --request-json)
      REQUEST_JSON="$2"; shift 2 ;;
    --output-json)
      OUTPUT_JSON="$2"; shift 2 ;;
    --scene-log)
      SCENE_LOG="$2"; shift 2 ;;
    --startup-wait)
      STARTUP_WAIT_SEC="$2"; shift 2 ;;
    --command)
      COMMAND="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64 ;;
  esac
done

if [[ "$MODE" != "dry_run" && "$MODE" != "ros_dry_run" ]]; then
  echo "ERROR: --mode must be dry_run or ros_dry_run" >&2
  exit 64
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
export PYTHONPATH

source_if_present() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$setup_file"
    set -u
  fi
}

source_if_present /opt/ros/jazzy/setup.bash
if [[ -f "$REPO_ROOT/external/kitchen_scene/install/setup.bash" ]]; then
  source_if_present "$REPO_ROOT/external/kitchen_scene/install/setup.bash"
else
  source_if_present "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/setup.bash"
fi

echo "[phase3a] repo: $REPO_ROOT"
echo "[phase3a] request: $REQUEST_JSON"
echo "[phase3a] output: $OUTPUT_JSON"
echo "[phase3a] mode: $MODE"

if [[ "$SKIP_BRIDGE_VALIDATION" -eq 0 ]]; then
  if [[ "$LAUNCH_SCENE" -eq 1 ]]; then
    echo "[phase3a] ensuring kitchen scene bridge"
    BRIDGE_CMD=(scripts/v5/bridge_kitchen_scene.sh)
  else
    echo "[phase3a] validating kitchen scene bridge"
    BRIDGE_CMD=(scripts/v5/bridge_kitchen_scene.sh --validate-only)
  fi
  printf '  %q' "${BRIDGE_CMD[@]}"; echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${BRIDGE_CMD[@]}"
  fi
else
  echo "[phase3a] skipping kitchen scene bridge validation"
fi

if [[ "$LAUNCH_SCENE" -eq 1 ]]; then
  LAUNCH_CMD=(scripts/v5/launch_kitchen_scene.sh --mode "$SCENE_MODE")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    LAUNCH_CMD+=(--dry-run)
  fi
  echo "[phase3a] launching kitchen scene"
  printf '  %q' "${LAUNCH_CMD[@]}"; echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$(dirname "$SCENE_LOG")"
    nohup "${LAUNCH_CMD[@]}" >"$SCENE_LOG" 2>&1 &
    echo "[phase3a] scene launch pid: $!"
    echo "[phase3a] scene log: $SCENE_LOG"
    sleep "$STARTUP_WAIT_SEC"
  else
    "${LAUNCH_CMD[@]}"
  fi
fi

if [[ ! -f "$REQUEST_JSON" ]]; then
  echo "[phase3a] request artifact missing; generating mock_qwen request"
  GEN_CMD=(python -m hrl_trainer.v5.qwen_l1_client --backend mock_qwen --command "$COMMAND" --output "$REQUEST_JSON")
  printf '  %q' "${GEN_CMD[@]}"; echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${GEN_CMD[@]}" >/dev/null
  fi
fi

RUN_CMD=(
  python -m hrl_trainer.v5.phase3a_runtime_node
  --request-json "$REQUEST_JSON"
  --output-json "$OUTPUT_JSON"
  --mode "$MODE"
)

echo "[phase3a] running runtime bridge"
printf '  %q' "${RUN_CMD[@]}"; echo
if [[ "$DRY_RUN" -eq 0 ]]; then
  "${RUN_CMD[@]}"
fi

echo "[phase3a] done"
