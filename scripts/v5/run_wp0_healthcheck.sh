#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--live] [--replay-bag PATH] [--output PATH] [--auto-launch-scene] [--bootstrap-scene] [--scene-wait SEC] [--dry-run]

Purpose:
  Run V5 WP0 unified healthcheck with standard config/artifacts paths.
  In --live mode, this script can auto-launch the kitchen scene and wait for required topics
  so behavior matches manual startup flow.

Options:
  --live                Enable live metrics mode.
  --replay-bag PATH     Replay bag path for rosbag replay checks.
  --output PATH         Output report path.
                        Default: $REPO_ROOT/artifacts/wp0/wp0_report.json
  --auto-launch-scene   In --live mode, auto-launch kitchen scene if required topics are missing.
  --bootstrap-scene     Build/source ENPM662 scene packages before checks (manual-flow parity).
  --scene-wait SEC      Wait timeout for required topics after launch.
                        Default: 40
  --dry-run             Print resolved command without execution.
  -h, --help            Show this help.
USAGE
}

LIVE=0
REPLAY_BAG=""
OUTPUT="$REPO_ROOT/artifacts/wp0/wp0_report.json"
DRY_RUN=0
AUTO_LAUNCH_SCENE=0
BOOTSTRAP_SCENE=1
SCENE_WAIT_SEC=40

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      LIVE=1
      shift
      ;;
    --replay-bag)
      REPLAY_BAG="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --auto-launch-scene)
      AUTO_LAUNCH_SCENE=1
      shift
      ;;
    --bootstrap-scene)
      BOOTSTRAP_SCENE=1
      shift
      ;;
    --scene-wait)
      SCENE_WAIT_SEC="$2"
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

CONFIG="$REPO_ROOT/hrl_ws/src/hrl_trainer/config/wp0_config.yaml"
ART_DIR="$REPO_ROOT/artifacts/wp0"
mkdir -p "$ART_DIR"

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

  echo "INFO: bootstrap scene workspace (colcon build ENPM662 packages under $scene_ws)"
  (
    cd "$scene_ws"
    PATH=/usr/bin:$PATH colcon build \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
      --packages-select kitchen_robot_description kitchen_robot_controller \
      --event-handlers console_direct+
  ) >"$ART_DIR/scene_colcon_build.log" 2>&1

  source_if_exists "$scene_setup_src"
  source_if_exists "$scene_setup_legacy"
}

wait_for_topic() {
  local topic="$1"
  local timeout_sec="$2"
  for ((i=0; i<timeout_sec; i++)); do
    if timeout 3 ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_topic_sample() {
  local topic="$1"
  local timeout_sec="$2"
  timeout "$timeout_sec" ros2 topic echo "$topic" --once >/dev/null 2>&1
}

require_topics_and_samples() {
  local timeout_sec="$1"
  local missing=0
  local required=(
    /clock
    /joint_states
    /v5/cam/overhead/rgb
    /v5/cam/side/rgb
  )
  for topic in "${required[@]}"; do
    if ! wait_for_topic "$topic" "$timeout_sec"; then
      echo "WARN: topic not ready: $topic" >&2
      missing=1
      continue
    fi
    if ! wait_for_topic_sample "$topic" "$timeout_sec"; then
      echo "WARN: topic has no sample yet: $topic" >&2
      missing=1
    fi
  done
  return $missing
}

kill_scene_processes() {
  # Kill launch umbrella + known child executables to avoid duplicate-node contamination.
  pkill -f 'ros2 launch kitchen_robot_description gazebo.launch.py' >/dev/null 2>&1 || true
  pkill -f 'ros_gz_sim create' >/dev/null 2>&1 || true
  pkill -f '/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge' >/dev/null 2>&1 || true
  pkill -f 'tray_pose_adapter_node' >/dev/null 2>&1 || true
  pkill -f 'object_id_publisher_node' >/dev/null 2>&1 || true
  pkill -f 'static_transform_publisher' >/dev/null 2>&1 || true
  pkill -f 'robot_state_publisher' >/dev/null 2>&1 || true
  pkill -f 'gz sim' >/dev/null 2>&1 || true
  sleep 2
}

SCENE_PID=""
cleanup() {
  if [[ -n "$SCENE_PID" ]] && kill -0 "$SCENE_PID" >/dev/null 2>&1; then
    kill "$SCENE_PID" >/dev/null 2>&1 || true
    sleep 1
  fi
}
trap cleanup EXIT

# Align environment with manual startup flow.
source_if_exists /opt/ros/jazzy/setup.bash
source_if_exists "$REPO_ROOT/install/setup.bash"
source_if_exists "$REPO_ROOT/external/ENPM662_Group4_FinalProject/src/install/setup.bash"
source_if_exists "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/setup.bash"

PY_RUN=(python3)
if [[ "$LIVE" -eq 1 ]]; then
  # ROS Jazzy on this machine is aligned with system Python 3.12 (not Homebrew 3.14).
  if [[ -x /usr/bin/python3.12 ]]; then
    PY_RUN=(/usr/bin/python3.12)
  fi
else
  if command -v uv >/dev/null 2>&1 && [[ -f "$REPO_ROOT/hrl_ws/pyproject.toml" ]]; then
    PY_RUN=(uv run --project hrl_ws python)
  fi
fi

CMD=("${PY_RUN[@]}" -m hrl_trainer.v5.tools.wp0_healthcheck --config "$CONFIG" --artifacts-dir "$ART_DIR" --output "$OUTPUT")
if [[ "$LIVE" -eq 1 ]]; then
  CMD+=(--live)
fi
if [[ -n "$REPLAY_BAG" ]]; then
  CMD+=(--replay-bag "$REPLAY_BAG")
fi

echo "Command: PYTHONPATH=$REPO_ROOT/hrl_ws/src/hrl_trainer ${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  if [[ "$LIVE" -eq 1 ]]; then
    echo "DRY-RUN: live mode topic precheck enabled"
    if [[ "$BOOTSTRAP_SCENE" -eq 1 ]]; then
      echo "DRY-RUN: would colcon build scene packages (kitchen_robot_description/kitchen_robot_controller)"
    fi
    if [[ "$AUTO_LAUNCH_SCENE" -eq 1 ]]; then
      echo "DRY-RUN: would auto launch scene via scripts/v5/launch_kitchen_scene.sh"
    fi
  fi
  echo "DRY-RUN: healthcheck skipped"
  exit 0
fi

if [[ "$LIVE" -eq 1 ]]; then
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: ros2 not found in PATH. Source ROS 2 first (e.g. /opt/ros/jazzy/setup.bash)." >&2
    exit 2
  fi

  bootstrap_scene_if_needed

  if [[ "$AUTO_LAUNCH_SCENE" -eq 1 ]]; then
    echo "INFO: resetting previous scene/bridge processes..."
    kill_scene_processes
    echo "INFO: auto-launching kitchen scene..."
    "$REPO_ROOT/scripts/v5/launch_kitchen_scene.sh" >"$ART_DIR/scene_launch.log" 2>&1 &
    SCENE_PID=$!
    if ! require_topics_and_samples "$SCENE_WAIT_SEC"; then
      echo "ERROR: required topics/samples still missing after auto-launch wait (${SCENE_WAIT_SEC}s)." >&2
      echo "See: $ART_DIR/scene_launch.log" >&2
      exit 3
    fi
    echo "INFO: required topics with live samples ready."
  else
    if ! require_topics_and_samples 6; then
      echo "ERROR: required live topics/samples missing."
      echo "Hint: manually run scripts/v5/launch_kitchen_scene.sh first, or rerun with --auto-launch-scene." >&2
      exit 3
    fi
  fi
fi

(
  cd "$REPO_ROOT"
  PYTHONPATH="hrl_ws/src/hrl_trainer:${PYTHONPATH:-}" "${CMD[@]}"
)
