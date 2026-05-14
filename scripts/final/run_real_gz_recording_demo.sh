#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-final_real_gz_gui_record_001}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/v5/phase3a_controlled_sim}"
RUN_DIR="$ARTIFACT_ROOT/$RUN_ID"
VIDEO_PATH="${VIDEO_PATH:-$REPO_ROOT/report/videos/real_gz_phase3a_controlled_sim.mp4}"
FFLOG="$RUN_DIR/ffmpeg_x11grab.log"

mkdir -p "$RUN_DIR" "$(dirname "$VIDEO_PATH")"

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

DISPLAY_VALUE="${DISPLAY:-:0.0}"
VIDEO_SIZE="${VIDEO_SIZE:-1280x720}"
FRAMERATE="${FRAMERATE:-15}"
RECORD_SECONDS="${RECORD_SECONDS:-90}"
MAX_TARGETS="${MAX_TARGETS:-3}"
TARGET_PROFILE="${TARGET_PROFILE:-visible_workspace}"
APPROACH_STEPS="${APPROACH_STEPS:-12}"
FINISHER_STEPS="${FINISHER_STEPS:-8}"
STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-18}"

cleanup_scene_processes() {
  pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py" >/dev/null 2>&1 || true
  pkill -f "gz sim.*v5_kitchen" >/dev/null 2>&1 || true
  pkill -f "gz sim.*worlds/v5_kitchen_empty.sdf" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ -n "${FFPID:-}" ]]; then
    kill -INT "$FFPID" >/dev/null 2>&1 || true
    wait "$FFPID" >/dev/null 2>&1 || true
  fi
  cleanup_scene_processes
}
trap cleanup EXIT

echo "[real-gz-demo] repo: $REPO_ROOT"
echo "[real-gz-demo] run id: $RUN_ID"
echo "[real-gz-demo] video: $VIDEO_PATH"
echo "[real-gz-demo] display: $DISPLAY_VALUE size=$VIDEO_SIZE fps=$FRAMERATE"

ffmpeg \
  -y \
  -f x11grab \
  -video_size "$VIDEO_SIZE" \
  -framerate "$FRAMERATE" \
  -i "$DISPLAY_VALUE" \
  -t "$RECORD_SECONDS" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  "$VIDEO_PATH" >"$FFLOG" 2>&1 &
FFPID=$!
echo "[real-gz-demo] ffmpeg pid: $FFPID"

bash scripts/v5/run_phase3a_controlled_sim.sh \
  --launch-scene \
  --scene-mode gui \
  --cleanup-scene \
  --startup-wait "$STARTUP_WAIT_SEC" \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --max-targets "$MAX_TARGETS" \
  --target-profile "$TARGET_PROFILE" \
  --approach-steps "$APPROACH_STEPS" \
  --finisher-steps "$FINISHER_STEPS" \
  --command-duration-s 0.40 \
  --settle-timeout-s 1.6

kill -INT "$FFPID" >/dev/null 2>&1 || true
wait "$FFPID" >/dev/null 2>&1 || true
unset FFPID

echo "[real-gz-demo] wrote video: $VIDEO_PATH"
echo "[real-gz-demo] wrote summary: $RUN_DIR/controlled_sim_summary.json"
echo "[real-gz-demo] wrote step log: $RUN_DIR/runtime_steps.jsonl"
