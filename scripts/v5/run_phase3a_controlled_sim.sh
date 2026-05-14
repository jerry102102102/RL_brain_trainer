#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/v5/phase3a_controlled_sim"
RUN_ID="controlled_sim_gui_$(date +%Y%m%d_%H%M%S)"
SCENE_LOG="$ARTIFACT_ROOT/kitchen_scene_gui_launch.log"
SCENE_MODE="gui"
LAUNCH_SCENE=0
CLEANUP_SCENE=0
STARTUP_WAIT_SEC=18
MAX_TARGETS=1
TARGET_PROFILE="smoke"
APPROACH_STEPS=""
FINISHER_STEPS=""
POLICY_DEVICE="cpu"
COMMAND_DURATION_S="0.35"
SETTLE_TIMEOUT_S="1.4"
TARGETS_JSON=""
REGISTRY=""
DRY_RUN=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--launch-scene] [--scene-mode gui|headless] [--max-targets N]

Purpose:
  Run a controlled Phase 3A Gazebo validation rollout.  The script can launch
  the existing kitchen scene, displays FK target markers in Gazebo, executes the
  frozen Approach -> Finisher policies through the existing ROS2/L3 path, and
  writes step-level logs.

Options:
  --launch-scene          Start kitchen scene through scripts/v5/launch_kitchen_scene.sh.
  --cleanup-scene         Cleanup kitchen scene processes after the controlled run.
  --scene-mode MODE       gui (default) or headless when --launch-scene is used.
  --startup-wait SEC      Seconds to wait after launching scene. Default: $STARTUP_WAIT_SEC.
  --artifact-root PATH    Output artifact root. Default: $ARTIFACT_ROOT.
  --run-id ID             Run directory name. Default: timestamped controlled_sim_gui_*.
  --targets-json PATH     Optional target list JSON.
  --registry PATH         Optional phase3a runtime model registry override.
  --max-targets N         Number of default FK targets if --targets-json is absent. Default: $MAX_TARGETS.
  --target-profile NAME   smoke or visible_workspace. Default: $TARGET_PROFILE.
  --approach-steps N      Override Approach rollout steps.
  --finisher-steps N      Override Finisher rollout steps.
  --policy-device DEVICE  SB3 device for policy inference. Default: cpu.
  --command-duration-s S  Joint trajectory time_from_start. Default: $COMMAND_DURATION_S.
  --settle-timeout-s S    Runtime settle/action timeout. Default: $SETTLE_TIMEOUT_S.
  --dry-run               Print commands only.
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launch-scene)
      LAUNCH_SCENE=1; shift ;;
    --cleanup-scene)
      CLEANUP_SCENE=1; shift ;;
    --scene-mode)
      SCENE_MODE="$2"; shift 2 ;;
    --startup-wait)
      STARTUP_WAIT_SEC="$2"; shift 2 ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --targets-json)
      TARGETS_JSON="$2"; shift 2 ;;
    --registry)
      REGISTRY="$2"; shift 2 ;;
    --max-targets)
      MAX_TARGETS="$2"; shift 2 ;;
    --target-profile)
      TARGET_PROFILE="$2"; shift 2 ;;
    --approach-steps)
      APPROACH_STEPS="$2"; shift 2 ;;
    --finisher-steps)
      FINISHER_STEPS="$2"; shift 2 ;;
    --policy-device)
      POLICY_DEVICE="$2"; shift 2 ;;
    --command-duration-s)
      COMMAND_DURATION_S="$2"; shift 2 ;;
    --settle-timeout-s)
      SETTLE_TIMEOUT_S="$2"; shift 2 ;;
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

if [[ "$SCENE_MODE" != "gui" && "$SCENE_MODE" != "headless" ]]; then
  echo "ERROR: --scene-mode must be gui or headless" >&2
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

source_if_present "$REPO_ROOT/hrl_ws/.venv/bin/activate"
source_if_present /opt/ros/jazzy/setup.bash
if [[ -f "$REPO_ROOT/external/kitchen_scene/install/setup.bash" ]]; then
  source_if_present "$REPO_ROOT/external/kitchen_scene/install/setup.bash"
else
  source_if_present "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/setup.bash"
fi

cleanup_scene_processes() {
  pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py" >/dev/null 2>&1 || true
  pkill -f "gz sim.*v5_kitchen" >/dev/null 2>&1 || true
}

mkdir -p "$ARTIFACT_ROOT"

echo "[phase3a-controlled] repo: $REPO_ROOT"
echo "[phase3a-controlled] artifact root: $ARTIFACT_ROOT"
echo "[phase3a-controlled] run id: $RUN_ID"

echo "[phase3a-controlled] ensuring kitchen scene bridge"
BRIDGE_CMD=(scripts/v5/bridge_kitchen_scene.sh)
printf '  %q' "${BRIDGE_CMD[@]}"; echo
if [[ "$DRY_RUN" -eq 0 ]]; then
  "${BRIDGE_CMD[@]}"
fi

if [[ "$LAUNCH_SCENE" -eq 1 ]]; then
  LAUNCH_CMD=(scripts/v5/launch_kitchen_scene.sh --mode "$SCENE_MODE")
  echo "[phase3a-controlled] launching kitchen scene ($SCENE_MODE)"
  printf '  %q' "${LAUNCH_CMD[@]}"; echo
  if [[ "$DRY_RUN" -eq 0 ]]; then
    nohup "${LAUNCH_CMD[@]}" >"$SCENE_LOG" 2>&1 &
    echo "[phase3a-controlled] scene launch pid: $!"
    echo "[phase3a-controlled] scene log: $SCENE_LOG"
    sleep "$STARTUP_WAIT_SEC"
  fi
fi

RUN_CMD=(
  python -m hrl_trainer.v5.phase3a_controlled_sim
  --artifact-root "$ARTIFACT_ROOT"
  --run-id "$RUN_ID"
  --max-targets "$MAX_TARGETS"
  --target-profile "$TARGET_PROFILE"
  --policy-device "$POLICY_DEVICE"
  --command-duration-s "$COMMAND_DURATION_S"
  --settle-timeout-s "$SETTLE_TIMEOUT_S"
)

if [[ -n "$TARGETS_JSON" ]]; then
  RUN_CMD+=(--targets-json "$TARGETS_JSON")
fi
if [[ -n "$REGISTRY" ]]; then
  RUN_CMD+=(--registry "$REGISTRY")
fi
if [[ -n "$APPROACH_STEPS" ]]; then
  RUN_CMD+=(--approach-steps "$APPROACH_STEPS")
fi
if [[ -n "$FINISHER_STEPS" ]]; then
  RUN_CMD+=(--finisher-steps "$FINISHER_STEPS")
fi

echo "[phase3a-controlled] running controlled sim rollout"
printf '  %q' "${RUN_CMD[@]}"; echo
if [[ "$DRY_RUN" -eq 0 ]]; then
  "${RUN_CMD[@]}"
fi

if [[ "$CLEANUP_SCENE" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  echo "[phase3a-controlled] cleaning up scene processes"
  cleanup_scene_processes
fi

echo "[phase3a-controlled] done"
