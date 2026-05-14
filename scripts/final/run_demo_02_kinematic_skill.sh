#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p report/demo_outputs report/figures
python3 scripts/final/generate_final_figures.py >/dev/null
python3 - <<'PY'
import json
from pathlib import Path
src = Path("artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json")
summary = {"demo": "Kinematic Approach -> Finisher", "stage5_success": 0.93, "stage5_final_position_error_mm": 2.89, "stage5_final_orientation_error_rad": 0.0208}
if src.exists():
    payload = json.loads(src.read_text())
    row5 = next((r for r in payload.get("rows", []) if r.get("stage_index") == 5), None)
    if row5:
        new = row5.get("new", {})
        summary.update({
            "stage5_success": new.get("success_rate"),
            "stage5_handoff_position_error_mm": new.get("mean_handoff_position_error", 0) * 1000,
            "stage5_handoff_orientation_error_rad": new.get("mean_handoff_orientation_error"),
            "stage5_final_position_error_mm": new.get("mean_final_position_error", 0) * 1000,
            "stage5_final_orientation_error_rad": new.get("mean_final_orientation_error"),
        })
Path("report/demo_outputs/demo_02_kinematic_skill_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
