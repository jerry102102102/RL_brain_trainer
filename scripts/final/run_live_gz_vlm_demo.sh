#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-local_skill}"
shift || true

COMMAND="${COMMAND:-Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.}"
RUN_ID="${RUN_ID:-live_demo_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/report/demo_outputs}"
HEADLESS="${HEADLESS:-0}"
LAUNCH_SCENE="${LAUNCH_SCENE:-0}"
CLEANUP_SCENE="${CLEANUP_SCENE:-0}"
STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-18}"
USE_QWEN="${USE_QWEN:-0}"
QWEN_BACKEND="${QWEN_BACKEND:-mock_qwen}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/final/run_live_gz_vlm_demo.sh [dry_run_l1|local_skill|tray_like_transport|route_prefix] [extra python args]

Environment:
  COMMAND="natural language command"
  RUN_ID="live_demo id"
  LAUNCH_SCENE=1|0
  HEADLESS=1|0
  CLEANUP_SCENE=1|0
  USE_QWEN=1|0
  QWEN_BACKEND=mock_qwen|qwen_subprocess

Examples:
  bash scripts/final/run_live_gz_vlm_demo.sh dry_run_l1
  bash scripts/final/run_live_gz_vlm_demo.sh local_skill
  bash scripts/final/run_live_gz_vlm_demo.sh tray_like_transport
  bash scripts/final/run_live_gz_vlm_demo.sh route_prefix
USAGE
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "dry_run_l1" && "$MODE" != "local_skill" && "$MODE" != "tray_like_transport" && "$MODE" != "route_prefix" ]]; then
  echo "ERROR: unsupported mode: $MODE" >&2
  usage >&2
  exit 64
fi

source_if_present() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$setup_file"
    set -u
  fi
}

cleanup_scene_processes() {
  pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py" >/dev/null 2>&1 || true
  pkill -f "gz sim.*v5_kitchen" >/dev/null 2>&1 || true
  pkill -f "gz sim.*worlds/v5_kitchen_empty.sdf" >/dev/null 2>&1 || true
  pkill -f "tray_pose_adapter_node" >/dev/null 2>&1 || true
  pkill -f "object_id_publisher_node" >/dev/null 2>&1 || true
  pkill -f "parameter_bridge" >/dev/null 2>&1 || true
  pkill -f "robot_state_publisher" >/dev/null 2>&1 || true
  pkill -f "controller_autobringup" >/dev/null 2>&1 || true
  sleep 1
}

wait_for_ros_topic_once() {
  local topic="$1"
  local timeout_s="$2"
  echo "[live-demo] waiting for topic: $topic (${timeout_s}s)"
  local deadline=$((SECONDS + timeout_s))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    if ros2 topic list 2>/dev/null | grep -Fx "$topic" >/dev/null 2>&1; then
      local topic_type
      topic_type="$(ros2 topic type "$topic" 2>/dev/null | head -1 || true)"
      if [[ -n "$topic_type" ]]; then
        timeout 8s ros2 topic echo --once "$topic" >/dev/null && return 0
      fi
    fi
    sleep 1
  done
  echo "ERROR: timed out waiting for topic data: $topic" >&2
  return 1
}

ensure_ros2_controllers() {
  local timeout_s="${CONTROLLER_BRINGUP_TIMEOUT:-90}"
  local bringup_script=""
  local candidate
  for candidate in \
    "$REPO_ROOT/external/kitchen_scene/install/kitchen_robot_description/share/kitchen_robot_description/launch/controller_autobringup.py" \
    "$REPO_ROOT/external/kitchen_scene/src/Kitchen_Robot_URDF.SLDASM/launch/controller_autobringup.py" \
    "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/kitchen_robot_description/share/kitchen_robot_description/launch/controller_autobringup.py"
  do
    if [[ -f "$candidate" ]]; then
      bringup_script="$candidate"
      break
    fi
  done

  if [[ -z "$bringup_script" ]]; then
    echo "WARN: controller_autobringup.py not found; continuing with scene launch defaults"
    return 0
  fi

  echo "[live-demo] ensuring ros2_control controllers are active (${timeout_s}s)"
  /usr/bin/python3 "$bringup_script" \
    --controller-manager /controller_manager \
    --timeout "$timeout_s" \
    --period 1.0 \
    --controllers joint_state_broadcaster arm_controller \
    >"$OUTPUT_ROOT/$RUN_ID/controller_autobringup_retry.log" 2>&1 || true
  tail -n 12 "$OUTPUT_ROOT/$RUN_ID/controller_autobringup_retry.log" || true
}

mkdir -p "$OUTPUT_ROOT/$RUN_ID"
PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
export PYTHONPATH

source_if_present "$REPO_ROOT/hrl_ws/.venv/bin/activate"
source_if_present /opt/ros/jazzy/setup.bash
if [[ -f "$REPO_ROOT/external/kitchen_scene/install/setup.bash" ]]; then
  source_if_present "$REPO_ROOT/external/kitchen_scene/install/setup.bash"
else
  source_if_present "$REPO_ROOT/external/ENPM662_Group4_FinalProject/install/setup.bash"
fi

echo "=============================================================================="
echo "LIVE GZ VLM DEMO"
echo "=============================================================================="
echo "mode: $MODE"
echo "run_id: $RUN_ID"
echo "command: $COMMAND"
echo "output: $OUTPUT_ROOT/$RUN_ID"
echo "launch_scene: $LAUNCH_SCENE"
if [[ "$LAUNCH_SCENE" != "1" && "$MODE" != "dry_run_l1" ]]; then
  echo "scene: using already-running Gazebo/ROS scene"
fi

if [[ "$LAUNCH_SCENE" == "1" && "$MODE" != "dry_run_l1" ]]; then
  echo
  echo "[1/6] Starting Gazebo scene"
  scripts/v5/bridge_kitchen_scene.sh
  cleanup_scene_processes
  SCENE_MODE="headless"
  if [[ "$HEADLESS" == "0" ]]; then
    SCENE_MODE="gui"
  fi
  nohup scripts/v5/launch_kitchen_scene.sh --mode "$SCENE_MODE" >"$OUTPUT_ROOT/$RUN_ID/kitchen_scene.log" 2>&1 &
  SCENE_PID=$!
  echo "$SCENE_PID" >"$OUTPUT_ROOT/$RUN_ID/kitchen_scene.pid"
  echo "[live-demo] scene pid: $SCENE_PID mode=$SCENE_MODE"
  sleep "$STARTUP_WAIT_SEC"
  ensure_ros2_controllers
  wait_for_ros_topic_once /joint_states 120
fi

if [[ "$MODE" == "dry_run_l1" ]]; then
  LAUNCH_SCENE=0
fi

USE_QWEN_FLAG="--no-use-qwen"
if [[ "$USE_QWEN" == "1" ]]; then
  USE_QWEN_FLAG="--use-qwen"
fi

echo
echo "[2/6] Running live L1 -> RL demo orchestrator"
python3 -m hrl_trainer.v5.demo_live_vlm_gz \
  --command "$COMMAND" \
  --demo-mode "$MODE" \
  "$USE_QWEN_FLAG" \
  --qwen-backend "$QWEN_BACKEND" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  "$@"

if [[ "$CLEANUP_SCENE" == "1" && "$LAUNCH_SCENE" == "1" ]]; then
  echo
  echo "[cleanup] stopping Gazebo scene processes"
  cleanup_scene_processes
fi

echo
echo "LIVE DEMO FINISHED"
echo "outputs: $OUTPUT_ROOT/$RUN_ID"
