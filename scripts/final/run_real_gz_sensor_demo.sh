#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
export PYTHONPATH

RUN_ID="${RUN_ID:-final_real_gz_sensor_demo_001}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/v5/phase3a_controlled_sim}"
RUN_DIR="$ARTIFACT_ROOT/$RUN_ID"
VIDEO_PATH="${VIDEO_PATH:-$REPO_ROOT/report/videos/real_gz_camera_phase3a_controlled_sim.mp4}"
VIDEO_SUMMARY="$REPO_ROOT/report/demo_outputs/demo_04_real_gz_camera_summary.json"
SPAWN_DEMO_CAMERA="${SPAWN_DEMO_CAMERA:-0}"
TOPIC_WAS_SET=0
if [[ -n "${TOPIC:-}" ]]; then
  TOPIC_WAS_SET=1
fi
TOPIC="${TOPIC:-/v5/cam/side/rgb}"
if [[ "$SPAWN_DEMO_CAMERA" == "1" && "$TOPIC_WAS_SET" != "1" ]]; then
  TOPIC="/v5/cam/demo/rgb"
fi
DEMO_CAMERA_SDF="${DEMO_CAMERA_SDF:-$REPO_ROOT/report/demo_assets/cam_demo.sdf}"
DEMO_CAMERA_X="${DEMO_CAMERA_X:-0.75}"
DEMO_CAMERA_Y="${DEMO_CAMERA_Y:--2.15}"
DEMO_CAMERA_Z="${DEMO_CAMERA_Z:-2.15}"
DEMO_CAMERA_R="${DEMO_CAMERA_R:-3.14159}"
DEMO_CAMERA_P="${DEMO_CAMERA_P:-0.72}"
DEMO_CAMERA_YAW="${DEMO_CAMERA_YAW:-1.92}"

mkdir -p "$RUN_DIR" "$REPO_ROOT/report/videos" "$REPO_ROOT/report/demo_outputs"

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
  echo "[real-gz-sensor-demo] waiting for topic: $topic (${timeout_s}s)"
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

cleanup() {
  if [[ -n "${REC_PID:-}" ]]; then
    kill -INT "$REC_PID" >/dev/null 2>&1 || true
    wait "$REC_PID" >/dev/null 2>&1 || true
  fi
  cleanup_scene_processes
}
trap cleanup EXIT

echo "[real-gz-sensor-demo] repo: $REPO_ROOT"
echo "[real-gz-sensor-demo] run id: $RUN_ID"
echo "[real-gz-sensor-demo] camera topic: $TOPIC"
echo "[real-gz-sensor-demo] video: $VIDEO_PATH"

scripts/v5/bridge_kitchen_scene.sh
cleanup_scene_processes
nohup scripts/v5/launch_kitchen_scene.sh --mode headless >"$RUN_DIR/kitchen_scene_headless.log" 2>&1 &
SCENE_PID=$!
echo "$SCENE_PID" >"$RUN_DIR/kitchen_scene_headless.pid"
echo "[real-gz-sensor-demo] scene pid: $SCENE_PID"
sleep "${STARTUP_WAIT_SEC:-18}"

wait_for_ros_topic_once /joint_states "${JOINT_STATE_WAIT_SEC:-75}"

if [[ "$SPAWN_DEMO_CAMERA" == "1" ]]; then
  echo "[real-gz-sensor-demo] spawning demo camera: $DEMO_CAMERA_SDF"
  ros2 run ros_gz_sim create \
    -name cam_demo \
    -file "$DEMO_CAMERA_SDF" \
    -x "$DEMO_CAMERA_X" \
    -y "$DEMO_CAMERA_Y" \
    -z "$DEMO_CAMERA_Z" \
    -R "$DEMO_CAMERA_R" \
    -P "$DEMO_CAMERA_P" \
    -Y "$DEMO_CAMERA_YAW" >"$RUN_DIR/cam_demo_spawn.log" 2>&1 || true
  ros2 run ros_gz_bridge parameter_bridge \
    /cam_demo/image@sensor_msgs/msg/Image[gz.msgs.Image] \
    --ros-args -r /cam_demo/image:=/v5/cam/demo/rgb >"$RUN_DIR/cam_demo_bridge.log" 2>&1 &
  DEMO_CAMERA_BRIDGE_PID=$!
  echo "$DEMO_CAMERA_BRIDGE_PID" >"$RUN_DIR/cam_demo_bridge.pid"
  sleep "${DEMO_CAMERA_WAIT_SEC:-3}"
fi

wait_for_ros_topic_once "$TOPIC" "${CAMERA_WAIT_SEC:-45}"

python3 scripts/final/record_gz_camera_topic.py \
  --topic "$TOPIC" \
  --output "$VIDEO_PATH" \
  --summary-json "$VIDEO_SUMMARY" \
  --duration "${RECORD_SECONDS:-100}" \
  --fps "${VIDEO_FPS:-10}" \
  --max-frames "${MAX_VIDEO_FRAMES:-1000}" \
  --warmup-timeout "${CAMERA_RECORDER_WARMUP_SEC:-30}" >"$RUN_DIR/camera_recorder.log" 2>&1 &
REC_PID=$!
echo "[real-gz-sensor-demo] recorder pid: $REC_PID"
sleep 2

python3 -m hrl_trainer.v5.phase3a_controlled_sim \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --max-targets "${MAX_TARGETS:-3}" \
  --target-profile "${TARGET_PROFILE:-visible_workspace}" \
  --policy-device "${POLICY_DEVICE:-cpu}" \
  --command-duration-s "${COMMAND_DURATION_S:-0.40}" \
  --settle-timeout-s "${SETTLE_TIMEOUT_S:-1.6}" \
  --approach-steps "${APPROACH_STEPS:-12}" \
  --finisher-steps "${FINISHER_STEPS:-8}"

kill -INT "$REC_PID" >/dev/null 2>&1 || true
wait "$REC_PID" || true
unset REC_PID

python3 - "$RUN_DIR" <<'PY'
import json
import pathlib
import statistics
import sys

run = pathlib.Path(sys.argv[1])
summary = json.loads((run / "controlled_sim_summary.json").read_text())
rows = [json.loads(line) for line in (run / "runtime_steps.jsonl").read_text().splitlines() if line.strip()]
compact = {k: summary[k] for k in [
    "run_id",
    "target_count",
    "success_rate",
    "handoff_confirmed_rate",
    "final_position_error_mean",
    "final_orientation_error_mean",
    "step_log_path",
]}
compact.update({
    "step_count": len(rows),
    "execution_ok_count": sum(bool(r["execution_ok"]) for r in rows),
    "command_paths": sorted(set(r["command_path"] for r in rows)),
    "mean_actual_joint_delta_l2": statistics.mean(r["actual_joint_delta_l2"] for r in rows) if rows else None,
    "max_actual_joint_delta_l2": max((r["actual_joint_delta_l2"] for r in rows), default=None),
    "mean_tracking_error_l2": statistics.mean(r["tracking_error_l2"] for r in rows) if rows else None,
})
out = pathlib.Path("report/demo_outputs/demo_04_real_gz_controlled_sim_summary.json")
out.write_text(json.dumps(compact, indent=2, sort_keys=True))
print(json.dumps(compact, indent=2, sort_keys=True))
PY

echo "[real-gz-sensor-demo] wrote camera video: $VIDEO_PATH"
echo "[real-gz-sensor-demo] wrote camera summary: $VIDEO_SUMMARY"
echo "[real-gz-sensor-demo] wrote runtime summary: $RUN_DIR/controlled_sim_summary.json"
