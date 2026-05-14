#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p report/demo_outputs report/figures
python3 scripts/final/generate_final_figures.py >/dev/null
python3 - <<'PY'
import json
from pathlib import Path
prefix = Path("artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/route_eval_sequential/route_eval_sequential_summary.json")
full = Path("artifacts/kinematic_phase1/route_curriculum/eval_prefix120_model_full483_001/route_eval_sequential_summary.json")
summary = {"demo": "Route curriculum", "baseline_success": 0.0435, "baseline_longest_prefix": 21, "prefix120_success": 1.0, "full483_success": 0.4741, "full483_longest_prefix": 170}
if prefix.exists():
    p = json.loads(prefix.read_text())
    summary.update({"prefix120_success": p.get("success_rate"), "prefix120_longest_prefix": p.get("longest_success_prefix"), "prefix120_distance_m": p.get("cumulative_successful_route_distance_m")})
if full.exists():
    f = json.loads(full.read_text())
    summary.update({"full483_success": f.get("success_rate"), "full483_longest_prefix": f.get("longest_success_prefix"), "first_failure_index": f.get("first_failure_index"), "first_failure_reason": f.get("first_failure_reason")})
Path("report/demo_outputs/demo_03_route_curriculum_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
