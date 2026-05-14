#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p report/videos report/demo_outputs
bash scripts/final/run_demo_01_qwen_bridge.sh
bash scripts/final/run_demo_02_kinematic_skill.sh
bash scripts/final/run_demo_03_route_curriculum.sh
python3 scripts/final/generate_final_videos.py
cat > report/demo_outputs/demo_run_summary.md <<'EOF'
# Final Demo Run Summary

Generated demo summaries:

- `report/demo_outputs/demo_01_qwen_bridge_output.json`
- `report/demo_outputs/demo_02_kinematic_skill_summary.json`
- `report/demo_outputs/demo_03_route_curriculum_summary.json`

Generated demo videos:

- `report/videos/demo_01_qwen_bridge.mp4`
- `report/videos/demo_02_kinematic_skill.mp4`
- `report/videos/demo_03_route_curriculum.mp4`
- `report/videos/final_demo_compilation.mp4`

These are headless MP4 videos generated from the official final-package outputs.
EOF
echo "Demo summaries generated in report/demo_outputs"
echo "Demo videos generated in report/videos"
