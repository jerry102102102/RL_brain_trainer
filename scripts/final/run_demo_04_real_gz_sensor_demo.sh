#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-final_real_gz_sensor_demo_005}"

echo "[demo04] Real Gazebo camera + RL controlled sim evidence"
echo "[demo04] Existing verified run id: $RUN_ID"
echo "[demo04] Video: report/videos/real_gz_camera_phase3a_controlled_sim.mp4"
echo "[demo04] Runtime summary: artifacts/v5/phase3a_controlled_sim/$RUN_ID/controlled_sim_summary.json"
echo "[demo04] Step log: artifacts/v5/phase3a_controlled_sim/$RUN_ID/runtime_steps.jsonl"
echo

if [[ "${RERUN_REAL_GZ_DEMO:-0}" == "1" ]]; then
  echo "[demo04] RERUN_REAL_GZ_DEMO=1, launching a fresh Gazebo run."
  RUN_ID="$RUN_ID" bash scripts/final/run_real_gz_sensor_demo.sh
else
  echo "[demo04] Using existing verified artifact. Set RERUN_REAL_GZ_DEMO=1 to rerun Gazebo."
fi

python3 - <<'PY'
import json
import pathlib

camera = pathlib.Path("report/demo_outputs/demo_04_real_gz_camera_summary.json")
runtime = pathlib.Path("report/demo_outputs/demo_04_real_gz_controlled_sim_summary.json")
print("[demo04] Camera summary:")
print(camera.read_text() if camera.exists() else "missing")
print("[demo04] Runtime summary:")
print(runtime.read_text() if runtime.exists() else "missing")
PY
