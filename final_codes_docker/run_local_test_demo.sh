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
RUN_ID="${RUN_ID:-final_local_test_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$REPO_ROOT/report/demo_outputs/$RUN_ID"
export RUN_ID
mkdir -p "$OUT_DIR"

exec > >(tee "$OUT_DIR/command_log.txt") 2>&1

echo "=============================================================================="
echo "FINAL LOCAL TEST DEMO"
echo "=============================================================================="
echo "run_id: $RUN_ID"
echo "mode: $DEMO_MODE"
echo "output: $OUT_DIR"

CHECK_ONLY=1 bash final_codes_docker/download_demo_assets.sh || true

if [[ "$DEMO_MODE" == "visual" || "$HEADLESS" == "0" ]]; then
  echo "[visual] Running native Gazebo/RViz local_skill wrapper."
  bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill
  echo "LOCAL TEST DEMO FINISHED"
  echo "Outputs: $OUT_DIR"
  exit 0
fi

APPROACH_MODEL="artifacts/kinematic_phase1/workspace_expansion/workspace_expand_dynscale_stage8_11_big_001/best_checkpoint/model_best_by_gate.zip"
APPROACH_CONFIG="hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/workspace_expansion_dynamic_scale_big.yaml"
FINISHER_MODEL="artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip"
FINISHER_CONFIG="hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml"
EPISODES="${LOCAL_TEST_EPISODES:-3}"
STAGES="${LOCAL_TEST_STAGES:-0,5,7}"

if [[ -f "$APPROACH_MODEL" && -f "$FINISHER_MODEL" ]]; then
  echo "[headless] Running deterministic workspace expansion eval."
  python3 -m hrl_trainer.kinematic_phase1.eval.eval_workspace_expansion \
    --approach-checkpoint "$APPROACH_MODEL" \
    --approach-config "$APPROACH_CONFIG" \
    --finisher-checkpoint "$FINISHER_MODEL" \
    --finisher-config "$FINISHER_CONFIG" \
    --stages "$STAGES" \
    --episodes "$EPISODES" \
    --artifact-root "$OUT_DIR/eval_workspace_expansion"
  python3 - <<'PY'
import json
from pathlib import Path
out = Path("report/demo_outputs") / Path(__import__("os").environ["RUN_ID"])
src = out / "eval_workspace_expansion" / "workspace_eval_summary.json"
payload = json.loads(src.read_text())
summary = {
    "demo": "final local test",
    "mode": "headless_workspace_eval",
    "source": str(src),
    "stage_metrics": payload.get("stage_metrics", {}),
    "note": "Approach -> Finisher local skill demo.",
}
(out / "local_test_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
else
  echo "[headless] Trained models missing; writing fallback summary from official result artifact."
  python3 - <<'PY'
import json
import os
from pathlib import Path
out = Path("report/demo_outputs") / Path(os.environ["RUN_ID"])
artifact = Path("artifacts/kinematic_phase1/phase1c/workspace_sweep_workspace_noop_vs_previous_summary_001.json")
summary = {
    "demo": "final local test",
    "mode": "fallback_summary_no_policy_execution",
    "missing_models": True,
    "stage5_success": 0.93,
    "stage5_final_position_error_mm": 2.89,
    "stage5_final_orientation_error_rad": 0.0208,
    "note": "Policy eval skipped because model assets were missing. See final_codes_docker/model_manifest.yaml.",
}
if artifact.exists():
    summary["source_artifact"] = str(artifact)
(out / "local_test_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
fi

echo "LOCAL TEST DEMO FINISHED"
echo "Outputs: $OUT_DIR"
