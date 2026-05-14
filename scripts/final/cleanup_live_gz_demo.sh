#!/usr/bin/env bash
set -euo pipefail

echo "[cleanup-live-demo] stopping Gazebo / ROS2 demo processes"
kill_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<<"$pids"
}
kill_pattern "ros2 launch kitchen_robot_description gazebo.launch.py"
kill_pattern "gz sim.*v5_kitchen"
kill_pattern "gz sim.*worlds/v5_kitchen_empty.sdf"
kill_pattern "tray_pose_adapter_node"
kill_pattern "object_id_publisher_node"
kill_pattern "parameter_bridge"
kill_pattern "robot_state_publisher"
kill_pattern "controller_autobringup"
kill_pattern "target_marker_node"
echo "[cleanup-live-demo] done"
