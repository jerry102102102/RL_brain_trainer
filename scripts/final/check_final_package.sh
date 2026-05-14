#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
status=0
check() {
  if [ -e "$1" ]; then echo "PASS $1"; else echo "FAIL $1"; status=1; fi
}
warn() {
  if [ -e "$1" ]; then echo "PASS $1"; else echo "WARN $1"; fi
}
check report/FINAL_PROJECT_SUMMARY.md
check report/OFFICIAL_ARTIFACTS.md
check report/FINAL_REPORT.md
check report/FINAL_REPORT.pdf
check report/FINAL_PRESENTATION.pptx
check report/DEMO_VIDEO_SCRIPT.md
check report/DEMO_RECORDING_COMMANDS.md
check report/REAL_GZ_DEMO_EVIDENCE.md
check report/figures/final_architecture_l1_l2_l3.png
check report/figures/workspace_sweep_stage_success.png
check report/figures/route_prefix_improvement.png
check report/figures/route_curriculum_limitations.png
check scripts/final/run_demo_01_qwen_bridge.sh
check scripts/final/run_demo_02_kinematic_skill.sh
check scripts/final/run_demo_03_route_curriculum.sh
check scripts/final/run_demo_04_real_gz_sensor_demo.sh
check scripts/final/run_live_gz_vlm_demo.sh
check scripts/final/run_live_gz_screen_recording_demo.sh
check scripts/final/cleanup_live_gz_demo.sh
check scripts/final/check_live_demo_ready.sh
check scripts/final/record_final_demo.sh
check scripts/final/generate_final_videos.py
check scripts/final/run_real_gz_sensor_demo.sh
check scripts/final/record_gz_camera_topic.py
check hrl_ws/src/hrl_trainer/hrl_trainer/v5/demo_live_vlm_gz.py
check hrl_ws/src/hrl_trainer/hrl_trainer/v5/target_marker_node.py
check config/rviz/phase3a_demo.rviz
warn report/videos/demo_01_qwen_bridge.mp4
warn report/videos/demo_02_kinematic_skill.mp4
warn report/videos/demo_03_route_curriculum.mp4
warn report/videos/final_demo_compilation.mp4
warn report/videos/real_gz_camera_phase3a_controlled_sim.mp4
warn report/demo_outputs/demo_04_real_gz_camera_summary.json
warn report/demo_outputs/demo_04_real_gz_controlled_sim_summary.json
warn report/demo_outputs/live_demo_local_skill_smoke_001/final_summary.json
warn report/demo_outputs/live_demo_local_skill_smoke_001/runtime_status.log
warn artifacts/v5/phase3a_controlled_sim/final_real_gz_sensor_demo_005/runtime_steps.jsonl
warn artifacts/v5/phase3a_controlled_sim/live_demo_local_skill_smoke_001_controlled_sim/runtime_steps.jsonl
warn artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json
warn artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json
warn artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip
if grep -q "FINAL_PROJECT_SUMMARY.md" README.md; then echo "PASS README final package links"; else echo "FAIL README final package links"; status=1; fi
if [ "$status" -eq 0 ]; then echo "FINAL PACKAGE CHECK: PASS"; else echo "FINAL PACKAGE CHECK: FAIL"; fi
exit "$status"
