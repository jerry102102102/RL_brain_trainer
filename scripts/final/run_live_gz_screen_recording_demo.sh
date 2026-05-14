#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-local_skill}"
shift || true

export HEADLESS="${HEADLESS:-0}"
export CLEANUP_SCENE="${CLEANUP_SCENE:-0}"
export LAUNCH_SCENE="${LAUNCH_SCENE:-0}"
export RUN_ID="${RUN_ID:-live_screen_recording_${MODE}_$(date +%Y%m%d_%H%M%S)}"
export START_RVIZ="${START_RVIZ:-0}"

EXTRA_ARGS=("$@")
if [[ "$MODE" == "local_skill" && "$#" -eq 0 ]]; then
  EXTRA_ARGS=(--max-targets 5 --target-profile recording_showcase --approach-steps 60 --finisher-steps 8 --marker-duration 360)
elif [[ "$MODE" == "tray_like_transport" && "$#" -eq 0 ]]; then
  EXTRA_ARGS=(--tray-like-waypoints 6 --tray-like-approach-steps 60 --tray-like-finisher-steps 8 --marker-duration 360)
fi

cat <<EOF
==============================================================================
LIVE GZ SCREEN RECORDING DEMO
==============================================================================
mode: $MODE
run_id: $RUN_ID
HEADLESS=$HEADLESS
CLEANUP_SCENE=$CLEANUP_SCENE
LAUNCH_SCENE=$LAUNCH_SCENE
START_RVIZ=$START_RVIZ

Recording tip:
  Put this terminal on the left and Gazebo/RViz on the right.
  Start Gazebo yourself first for the cleanest recording:
    source /opt/ros/jazzy/setup.zsh
    source external/ENPM662_Group4_FinalProject/install/setup.zsh
    ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
  This script defaults to LAUNCH_SCENE=0, so it only runs the L1/L2/L3 demo.
  Recommended demos:
    bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill
    bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport
  When finished, run:
    bash scripts/final/cleanup_live_gz_demo.sh
==============================================================================
EOF

if [[ "$START_RVIZ" == "1" ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  if [[ -f external/ENPM662_Group4_FinalProject/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source external/ENPM662_Group4_FinalProject/install/setup.bash
  elif [[ -f external/kitchen_scene/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source external/kitchen_scene/install/setup.bash
  fi
  set -u
  echo "[screen-demo] starting RViz with camera + marker config"
  nohup rviz2 -d config/rviz/phase3a_demo.rviz >"report/demo_outputs/${RUN_ID}_rviz.log" 2>&1 &
  echo "$!" >"report/demo_outputs/${RUN_ID}_rviz.pid"
fi

bash scripts/final/run_live_gz_vlm_demo.sh "$MODE" "${EXTRA_ARGS[@]}"
