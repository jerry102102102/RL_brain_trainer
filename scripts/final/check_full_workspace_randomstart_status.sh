#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-workspace_full_coverage_randomstart_overnight_001}"
RUN_DIR="artifacts/kinematic_phase1/workspace_full_coverage_randomstart/$RUN_ID"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "FAIL: run directory does not exist: $RUN_DIR"
  exit 1
fi

echo "=============================================================================="
echo "FULL WORKSPACE RANDOM-START STATUS"
echo "=============================================================================="
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"

if [[ -f "$RUN_DIR/run.pid" ]]; then
  PID="$(cat "$RUN_DIR/run.pid")"
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "process: RUNNING pid=$PID"
  else
    echo "process: not running pid=$PID"
  fi
fi
if [[ -f "$RUN_DIR/run.tmux" ]]; then
  echo "tmux: $(cat "$RUN_DIR/run.tmux")"
fi
if [[ -f "$RUN_DIR/best_checkpoint/model_best_by_gate.zip" ]]; then
  echo "best_by_gate: $RUN_DIR/best_checkpoint/model_best_by_gate.zip"
else
  echo "best_by_gate: not saved yet"
fi
if [[ -f "$RUN_DIR/latest_checkpoint/model_latest.zip" ]]; then
  echo "latest: $RUN_DIR/latest_checkpoint/model_latest.zip"
fi

python3 - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
def load(path):
    return json.loads(path.read_text()) if path.exists() else None
selection = load(run / "best_model_selection_summary.json")
if selection:
    print("\nGate selection:")
    for key in ("score", "retention_ok", "highest_passed_stage", "score_stage_index"):
        print(f"  {key}: {selection.get(key)}")
metrics = load(run / "stage_metrics.json")
if metrics is None:
    eval_dirs = sorted((run / "gate_evals").glob("eval_step_*")) if (run / "gate_evals").exists() else []
    if eval_dirs:
        metrics = load(eval_dirs[-1] / "stage_metrics.json")
if metrics:
    print("\nHome-start stage table:")
    for key in sorted(metrics, key=lambda v: int(v)):
        row = metrics[key]
        print(
            f"  stage {int(key):02d}: success={row.get('success_rate', 0):.3f} "
            f"ready={row.get('finisher_ready_hit_rate', 0):.3f} "
            f"pos={row.get('mean_final_position_error', 0):.5f} "
            f"ori={row.get('mean_final_orientation_error', 0):.5f}"
        )
coverage = load(run / "final_full_workspace_eval" / "full_workspace_coverage_summary.json")
if coverage is None:
    coverage = load(run / "full_workspace_coverage_summary.json")
if coverage:
    print("\nRandom-start coverage:")
    print("  known success:", coverage.get("random_start_known_workspace", {}).get("success_rate"))
    print("  frontier success:", coverage.get("random_start_frontier", {}).get("success_rate"))
    print("  stress success:", coverage.get("full_reachable_stress", {}).get("success_rate"))
    print("  covered bucket fraction:", coverage.get("covered_bucket_fraction"))
    print("  stable bucket fraction:", coverage.get("stable_bucket_fraction"))
    print("  partial bucket fraction:", coverage.get("partial_bucket_fraction"))
hist = run / "eval_history.jsonl"
if hist.exists():
    lines = [line for line in hist.read_text().splitlines() if line.strip()]
    print(f"\neval_history records: {len(lines)}")
    for line in lines[-3:]:
        print("  " + line[:500])
PY

echo
echo "Log tail:"
if [[ -f "$RUN_DIR/run.log" ]]; then
  tail -60 "$RUN_DIR/run.log"
else
  echo "WARN: no run.log yet"
fi
