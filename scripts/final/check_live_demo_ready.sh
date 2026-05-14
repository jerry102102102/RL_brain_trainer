#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

status=0

pass() { echo "PASS $*"; }
warn() { echo "WARN $*"; }
fail() { echo "FAIL $*"; status=1; }

exists() {
  if [[ -e "$1" ]]; then pass "$1"; else fail "$1"; fi
}

optional_exists() {
  if [[ -e "$1" ]]; then pass "$1"; else warn "$1"; fi
}

cmd_exists() {
  if command -v "$1" >/dev/null 2>&1; then pass "command:$1"; else fail "command:$1"; fi
}

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
fi

cmd_exists python3
cmd_exists gz
cmd_exists ros2

exists scripts/final/run_live_gz_vlm_demo.sh
exists hrl_ws/src/hrl_trainer/hrl_trainer/v5/demo_live_vlm_gz.py
exists hrl_ws/src/hrl_trainer/hrl_trainer/v5/target_marker_node.py
exists hrl_ws/src/hrl_trainer/hrl_trainer/v5/configs/phase3a_runtime_models.yaml
exists artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
exists artifacts/kinematic_phase1/phase1c/approach_finisher_ready_visible_workspace_ft_3m9_001/model_latest.zip
exists artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip
exists artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
optional_exists config/rviz/phase3a_demo.rviz

if ros2 topic list >/tmp/live_demo_topics.txt 2>/dev/null; then
  if grep -Fx "/joint_states" /tmp/live_demo_topics.txt >/dev/null; then pass "topic:/joint_states"; else warn "topic:/joint_states not currently active"; fi
  if grep -Fx "/v5/perception/object_pose_est" /tmp/live_demo_topics.txt >/dev/null; then pass "topic:/v5/perception/object_pose_est"; else warn "topic:/v5/perception/object_pose_est not currently active"; fi
  if grep -Fx "/v5/demo/target_marker" /tmp/live_demo_topics.txt >/dev/null; then pass "topic:/v5/demo/target_marker"; else warn "topic:/v5/demo/target_marker not currently active until demo starts"; fi
else
  warn "ros2 topic list unavailable; start/source ROS2 scene before live demo"
fi

if ros2 service list >/tmp/live_demo_services.txt 2>/dev/null; then
  if grep -Fx "/controller_manager/list_controllers" /tmp/live_demo_services.txt >/dev/null; then
    pass "service:/controller_manager/list_controllers"
  else
    warn "service:/controller_manager/list_controllers not currently active"
  fi
else
  warn "ros2 service list unavailable; start/source ROS2 scene before live demo"
fi

if [[ "$status" -eq 0 ]]; then
  echo "LIVE DEMO CHECK: PASS"
else
  echo "LIVE DEMO CHECK: FAIL"
fi
exit "$status"
