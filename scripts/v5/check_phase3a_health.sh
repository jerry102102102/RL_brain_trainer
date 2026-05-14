#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMEOUT_SEC=5

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--timeout SEC]

Purpose:
  Check the Phase 3A Gazebo/L1-L2-L3 runtime surface.

Checks:
  - ROS2 CLI availability
  - /controller_manager/list_controllers
  - /tray1/pose
  - /v5/perception/object_pose_est
  - /v5/intent_packet
  - /v5/skill_command
  - /joint_states
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)
      TIMEOUT_SEC="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64 ;;
  esac
done

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

if ! command -v ros2 >/dev/null 2>&1; then
  echo "FAIL: ros2 CLI not found. Source ROS2 Jazzy and scene install setup first." >&2
  exit 2
fi

echo "[phase3a-health] controller list"
if ! timeout "$TIMEOUT_SEC" ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers "{}" >/tmp/phase3a_controllers.log 2>&1; then
  echo "WARN: controller_manager list_controllers did not respond within ${TIMEOUT_SEC}s"
else
  grep -E "arm_controller|joint_state_broadcaster" /tmp/phase3a_controllers.log || true
fi

check_topic_once() {
  local topic="$1"
  local qos="${2:-}"
  echo "[phase3a-health] sample topic: $topic"
  if [[ -n "$qos" ]]; then
    timeout "$TIMEOUT_SEC" ros2 topic echo "$topic" --qos-reliability "$qos" --once >/tmp/phase3a_topic.log 2>&1
  else
    timeout "$TIMEOUT_SEC" ros2 topic echo "$topic" --once >/tmp/phase3a_topic.log 2>&1
  fi
}

check_topic_once /joint_states || echo "WARN: no /joint_states sample"
check_topic_once /tray1/pose best_effort || echo "WARN: no /tray1/pose sample"
check_topic_once /v5/perception/object_pose_est best_effort || echo "WARN: no /v5/perception/object_pose_est sample"

echo "[phase3a-health] topic list contract surface"
ros2 topic list | grep -E '^/v5/intent_packet$|^/v5/skill_command$|^/arm_controller/joint_trajectory$|^/joint_states$|^/v5/perception/object_pose_est$' || true

echo "[phase3a-health] complete"
