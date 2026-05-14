#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p report/demo_outputs
python3 - <<'PY'
import json
from pathlib import Path
src = Path("artifacts/v5/qwen_l1_demo/l1_to_rl_skill_request_qwen.json")
if not src.exists():
    print("WARN: Qwen artifact missing; using fallback summary.")
    payload = {"tool_call":{"tool":"resolve_intent_packet","arguments":{"object_id":"tray1","source_slot":"shelf_A1","target_slot":"shelf_B1","constraints":{"speed_cap":"SLOW"}}},"skill_request":{"pipeline":"APPROACH -> FINISHER","target_pose":{"xyz":[-0.92,-1.16,1.22],"rpy":[3.14,0.0,3.14]}}}
else:
    payload = json.loads(src.read_text())
skill = payload.get("skill_request", {})
summary = {
    "demo": "Qwen L1 bridge",
    "tool": payload.get("tool_call", {}).get("tool"),
    "object_id": skill.get("object_id", "tray1"),
    "source_slot": skill.get("source_slot", "shelf_A1"),
    "target_slot": skill.get("target_slot", "shelf_B1"),
    "pipeline": skill.get("pipeline", "APPROACH -> FINISHER"),
    "target_pose": skill.get("target_pose", {"xyz":[-0.92,-1.16,1.22],"rpy":[3.14,0.0,3.14]}),
    "safety_boundary": "L1 semantic only; no raw joint actions.",
}
out = Path("report/demo_outputs/demo_01_qwen_bridge_output.json")
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
