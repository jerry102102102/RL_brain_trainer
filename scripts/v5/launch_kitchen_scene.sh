#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_SCENE_LINK="$REPO_ROOT/external/ENPM662_Group4_FinalProject"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--scene-link PATH] [--launch-cmd CMD] [--dry-run]

Purpose:
  Launch kitchen scene from the bridged ENPM662 repo.

Options:
  --scene-link PATH   Path to bridged scene repo symlink.
                      Default: $DEFAULT_SCENE_LINK
  --launch-cmd CMD    Explicit launch command run inside scene repo.
                      Example: ros2 launch my_pkg kitchen_scene.launch.py use_sim_time:=true
  --dry-run           Print launch command only.
  -h, --help          Show this help.
USAGE
}

SCENE_LINK="$DEFAULT_SCENE_LINK"
LAUNCH_CMD=""
DRY_RUN=0

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

if [[ ! -d "$SCENE_LINK" ]]; then
  echo "ERROR: bridged scene path not found: $SCENE_LINK" >&2
  echo "Run: scripts/v5/bridge_kitchen_scene.sh --scene-repo <path>" >&2
  exit 2
fi

if [[ -z "$LAUNCH_CMD" ]]; then
  if command -v rg >/dev/null 2>&1; then
    CANDIDATE="$(rg --files "$SCENE_LINK" | rg 'launch/.*(kitchen|gazebo).*\.launch\.py$|(kitchen|gazebo).*\.launch\.py$' | head -n 1 || true)"
  else
    CANDIDATE="$(find "$SCENE_LINK" -type f \( -name '*kitchen*.launch.py' -o -name '*gazebo*.launch.py' \) | head -n 1 || true)"
  fi

  if [[ -z "$CANDIDATE" ]]; then
    cat >&2 <<ERR
ERROR: no kitchen launch file auto-detected under $SCENE_LINK
Provide explicit command with:
  --launch-cmd 'ros2 launch <pkg> <launch>.launch.py use_sim_time:=true'
ERR
    exit 3
  fi

  PKG_DIR="$(basename "$(dirname "$(dirname "$CANDIDATE")")")"
  LAUNCH_FILE="$(basename "$CANDIDATE")"
  LAUNCH_CMD="ros2 launch $PKG_DIR $LAUNCH_FILE use_sim_time:=true headless:=true"
fi

echo "Scene repo: $SCENE_LINK"
echo "Launch cmd: $LAUNCH_CMD"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: launch skipped"
  exit 0
fi

(
  cd "$SCENE_LINK"
  bash -lc "$LAUNCH_CMD"
)
