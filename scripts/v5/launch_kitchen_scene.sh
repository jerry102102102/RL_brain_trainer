#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_SCENE_LINK="$REPO_ROOT/external/kitchen_scene"
LEGACY_SCENE_LINK="$REPO_ROOT/external/ENPM662_Group4_FinalProject"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--scene-link PATH] [--launch-cmd CMD] [--mode headless|gui] [--no-cleanup] [--dry-run]

Purpose:
  Launch kitchen scene from the bridged ENPM662 repo.

Options:
  --scene-link PATH   Path to bridged scene repo symlink.
                      Default: $DEFAULT_SCENE_LINK
  --launch-cmd CMD    Explicit launch command run inside scene repo.
                      Example: ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=true
  --mode MODE         Auto launch mode when --launch-cmd is not set.
                      headless (default) or gui
  --no-cleanup        Skip pre-launch cleanup of existing kitchen scene processes.
  --dry-run           Print launch command only.
  -h, --help          Show this help.
USAGE
}

SCENE_LINK="$DEFAULT_SCENE_LINK"
LAUNCH_CMD=""
MODE="headless"
DRY_RUN=0
DO_CLEANUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-link)
      SCENE_LINK="$2"
      shift 2
      ;;
    --launch-cmd)
      LAUNCH_CMD="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --no-cleanup)
      DO_CLEANUP=0
      shift
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

if [[ "$MODE" != "headless" && "$MODE" != "gui" ]]; then
  echo "ERROR: --mode must be one of: headless, gui" >&2
  exit 64
fi

if [[ "$SCENE_LINK" == "$DEFAULT_SCENE_LINK" && ! -d "$SCENE_LINK" && -d "$LEGACY_SCENE_LINK" ]]; then
  echo "WARN: default bridge not found at $DEFAULT_SCENE_LINK, fallback to legacy path: $LEGACY_SCENE_LINK" >&2
  SCENE_LINK="$LEGACY_SCENE_LINK"
fi

if [[ ! -d "$SCENE_LINK" ]]; then
  echo "ERROR: bridged scene path not found: $SCENE_LINK" >&2
  echo "Run: scripts/v5/bridge_kitchen_scene.sh --scene-repo <path>" >&2
  exit 2
fi


cleanup_scene_processes() {
  local patterns=(
    "ros2 launch kitchen_robot_description gazebo.launch.py"
    "gz sim.*worlds/v5_kitchen_empty.sdf"
    "gz sim.*world=v5_kitchen_empty"
  )

  local killed_any=0
  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "Cleanup: terminating existing process pattern: $pattern"
      pkill -f "$pattern" >/dev/null 2>&1 || true
      killed_any=1
    fi
  done

  if [[ "$killed_any" -eq 1 ]]; then
    sleep 1
  fi
}

if [[ -z "$LAUNCH_CMD" ]]; then
  if [[ "$MODE" == "headless" ]]; then
    LAUNCH_CMD="ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=true use_software_renderer:=true"
  else
    LAUNCH_CMD="ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false"
  fi
fi

echo "Scene repo: $SCENE_LINK"
echo "Launch mode: $MODE"
echo "Cleanup before launch: $([[ "$DO_CLEANUP" -eq 1 ]] && echo enabled || echo disabled)"
echo "Launch cmd: $LAUNCH_CMD"

if [[ "$DO_CLEANUP" -eq 1 ]]; then
  cleanup_scene_processes
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: launch skipped"
  exit 0
fi

(
  cd "$SCENE_LINK"
  bash -lc "$LAUNCH_CMD"
)
