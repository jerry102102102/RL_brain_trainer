#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_SCENE_ROOT="$REPO_ROOT/external/ENPM662_Group4_FinalProject"
LOG_ROOT="$REPO_ROOT/artifacts/wp0/smoke_runs"

SCENE_ROOT="$DEFAULT_SCENE_ROOT"
LAUNCH_WAIT_SEC=20
TOPIC_WAIT_SEC=40
DRY_RUN=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--scene-root PATH] [--launch-wait SEC] [--topic-wait SEC] [--dry-run]

Purpose:
  Minimal WP0 smoke on vendored ENPM662 scene inside RL_brain_trainer.
  Steps: build -> launch (headless) -> topic checks -> control trigger.

Options:
  --scene-root PATH   Scene project root.
                      Default: $DEFAULT_SCENE_ROOT
  --launch-wait SEC   Wait after launch before topic checks.
                      Default: $LAUNCH_WAIT_SEC
  --topic-wait SEC    Per-topic timeout for readiness checks.
                      Default: $TOPIC_WAIT_SEC
  --dry-run           Print resolved commands only.
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-root)
      SCENE_ROOT="$2"
      shift 2
      ;;
    --launch-wait)
      LAUNCH_WAIT_SEC="$2"
      shift 2
      ;;
    --topic-wait)
      TOPIC_WAIT_SEC="$2"
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

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_TS"
LAUNCH_LOG="$RUN_DIR/launch.log"
RESET_LOG="$RUN_DIR/joint_reset.log"
TRAJ_LOG="$RUN_DIR/traj_once.log"

mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$1"
}

source_if_exists() {
  local setup_file="$1"
  if [[ ! -f "$setup_file" ]]; then
    log "ERROR: missing setup file: $setup_file"
    return 1
  fi
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

wait_for_any_topic() {
  local timeout_sec="$1"
  shift
  local topic
  for ((i=0; i<timeout_sec; i++)); do
    for topic in "$@"; do
      if timeout 3 ros2 topic list 2>/dev/null | grep -qx "$topic"; then
        echo "$topic"
        return 0
      fi
    done
    sleep 1
  done
  return 1
}

cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "$LAUNCH_PID" >/dev/null 2>&1; then
    kill "$LAUNCH_PID" >/dev/null 2>&1 || true
    sleep 2
  fi
  pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -d "$SCENE_ROOT" ]]; then
  log "ERROR: scene root missing: $SCENE_ROOT"
  exit 2
fi
if [[ ! -d "$SCENE_ROOT/src" || ! -d "$SCENE_ROOT/scripts" || ! -f "$SCENE_ROOT/README.md" ]]; then
  log "ERROR: scene root missing required content (src/scripts/README.md): $SCENE_ROOT"
  exit 2
fi

log "WP0 scene smoke start"
log "Scene root: $SCENE_ROOT"
log "Logs: $RUN_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN build: colcon build --base-paths \"$SCENE_ROOT/src\" --build-base \"$SCENE_ROOT/build\" --install-base \"$SCENE_ROOT/install\" --packages-select kitchen_robot_description kitchen_robot_controller --event-handlers console_direct+"
  echo "DRY-RUN launch: ros2 launch kitchen_robot_description gazebo.launch.py headless:=true"
  echo "DRY-RUN control: ros2 run kitchen_robot_controller joint_reset_node --ros-args -p publish_repeat_count:=3 -p publish_repeat_period:=0.2 -p time_to_reach:=1.0"
  exit 0
fi

source_if_exists /opt/ros/jazzy/setup.bash

log "Build scene packages"
colcon build \
  --base-paths "$SCENE_ROOT/src" \
  --build-base "$SCENE_ROOT/build" \
  --install-base "$SCENE_ROOT/install" \
  --packages-select kitchen_robot_description kitchen_robot_controller \
  --event-handlers console_direct+ >"$RUN_DIR/colcon_build.log" 2>&1

source_if_exists "$SCENE_ROOT/install/setup.bash"

export ROS_LOG_DIR="$RUN_DIR/ros_logs"
mkdir -p "$ROS_LOG_DIR"
export ROS_HOME="$RUN_DIR/ros_home"
mkdir -p "$ROS_HOME/locks"

pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py" >/dev/null 2>&1 || true

log "Launch scene headless"
ros2 launch kitchen_robot_description gazebo.launch.py headless:=true >"$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!

sleep "$LAUNCH_WAIT_SEC"

REQUIRED_TOPICS=(
  /clock
  /joint_states
  /arm_controller/joint_trajectory
  /v5/cam/overhead/rgb
  /v5/cam/side/rgb
)

for topic in "${REQUIRED_TOPICS[@]}"; do
  log "Check topic: $topic"
  if ! wait_for_topic "$topic" "$TOPIC_WAIT_SEC"; then
    log "ERROR: topic not ready within ${TOPIC_WAIT_SEC}s: $topic"
    exit 1
  fi
done

log "Check tray tracking topic (accept raw or normalized stream)"
TRAY_TOPIC="$(wait_for_any_topic "$TOPIC_WAIT_SEC" /tray_tracking/pose_stream_raw /tray_tracking/pose_stream || true)"
if [[ -z "$TRAY_TOPIC" ]]; then
  log "ERROR: tray topic not ready within ${TOPIC_WAIT_SEC}s: /tray_tracking/pose_stream_raw or /tray_tracking/pose_stream"
  exit 1
fi
log "PASS: tray topic ready: $TRAY_TOPIC"

log "Trigger control via joint_reset_node"
timeout 12 ros2 topic echo /arm_controller/joint_trajectory --once >"$TRAJ_LOG" 2>&1 &
ECHO_PID=$!

timeout 20 ros2 run kitchen_robot_controller joint_reset_node \
  --ros-args \
  -p publish_repeat_count:=3 \
  -p publish_repeat_period:=0.2 \
  -p time_to_reach:=1.0 >"$RESET_LOG" 2>&1 || true

wait "$ECHO_PID" || true

if command -v rg >/dev/null 2>&1; then
  if rg -q "^joint_names:" "$TRAJ_LOG" && rg -q "^points:" "$TRAJ_LOG"; then
    log "PASS: trajectory observed on /arm_controller/joint_trajectory"
  else
    log "ERROR: no trajectory payload observed; see $TRAJ_LOG"
    exit 1
  fi
else
  if grep -q "^joint_names:" "$TRAJ_LOG" && grep -q "^points:" "$TRAJ_LOG"; then
    log "PASS: trajectory observed on /arm_controller/joint_trajectory"
  else
    log "ERROR: no trajectory payload observed; see $TRAJ_LOG"
    exit 1
  fi
fi

log "WP0 scene smoke PASS"
