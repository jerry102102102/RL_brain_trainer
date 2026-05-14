#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/hrl_ws/.venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hrl_ws/.venv/bin/activate"
  set -u
fi

export PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
export HEADLESS="${HEADLESS:-1}"
DEMO_MODE="${DEMO_MODE:-headless}"
RUN_ID="${RUN_ID:-final_full_route_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$REPO_ROOT/report/demo_outputs/$RUN_ID"
export RUN_ID
mkdir -p "$OUT_DIR"

exec > >(tee "$OUT_DIR/command_log.txt") 2>&1

echo "=============================================================================="
echo "FINAL FULL ROUTE / ROUTE-PREFIX DEMO"
echo "=============================================================================="
echo "run_id: $RUN_ID"
echo "mode: $DEMO_MODE"
echo "output: $OUT_DIR"
echo "claim: route-curriculum evidence, not full holder1-to-holder8 completion"

CHECK_ONLY=1 bash final_codes_docker/download_demo_assets.sh || true

if [[ "$DEMO_MODE" == "visual" || "$HEADLESS" == "0" ]]; then
  echo "[visual] Running native route_prefix wrapper."
  bash scripts/final/run_live_gz_vlm_demo.sh route_prefix
  echo "FULL ROUTE / ROUTE-PREFIX DEMO FINISHED"
  echo "Outputs: $OUT_DIR"
  exit 0
fi

ROUTE_MODEL="artifacts/kinematic_phase1/route_curriculum/route_prefix120_routeobs_sequence2_1m_001/model_latest.zip"
ROUTE_CONFIG="hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/route_curriculum_prefix120_routeobs_sequence2.yaml"
ROUTE_PATH="artifacts/kinematic_phase1/phase1c/scene_route_curriculum/tray1_holder1_to_8_route_q_dense.json"
END_INDEX="${FULL_ROUTE_END_INDEX:-120}"

if [[ -f "$ROUTE_MODEL" && -f "$ROUTE_PATH" ]]; then
  echo "[headless] Running route sequential eval through waypoint $END_INDEX."
  python3 -m hrl_trainer.kinematic_phase1.eval.eval_route_curriculum \
    --checkpoint "$ROUTE_MODEL" \
    --config "$ROUTE_CONFIG" \
    --route-path "$ROUTE_PATH" \
    --start-index 1 \
    --end-index "$END_INDEX" \
    --artifact-root "$OUT_DIR/route_eval"
  cp "$OUT_DIR/route_eval/route_eval_sequential_summary.json" "$OUT_DIR/route_demo_summary.json"
else
  echo "[headless] Route model or route artifact missing; writing fallback summary."
  python3 - <<'PY'
import json
import os
from pathlib import Path
out = Path("report/demo_outputs") / Path(os.environ["RUN_ID"])
fallback = Path("artifacts/kinematic_phase1/route_curriculum/eval_prefix120_model_full483_001/route_eval_sequential_summary.json")
summary = {
    "demo": "final full route / route-prefix",
    "mode": "fallback_summary_no_policy_execution",
    "missing_models_or_route": True,
    "baseline_longest_prefix": 21,
    "prefix120_success": 1.0,
    "full483_probe_success": 0.4741,
    "full483_longest_prefix": 170,
    "note": "Policy route eval skipped because model/route assets were missing. This is route-curriculum evidence, not full transport completion.",
}
if fallback.exists():
    summary.update(json.loads(fallback.read_text()))
(out / "route_demo_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
fi

cat > "$OUT_DIR/route_result_explanation.txt" <<'EOF'
FULL ROUTE / ROUTE-PREFIX DEMO FINISHED
This is route-curriculum evidence, not full transport completion.
The key metric is longest successful prefix / waypoint progress, not "full route solved".
EOF

cat "$OUT_DIR/route_demo_summary.json"
echo
echo "FULL ROUTE / ROUTE-PREFIX DEMO FINISHED"
echo "This is route-curriculum evidence, not full transport completion."
echo "Outputs: $OUT_DIR"
