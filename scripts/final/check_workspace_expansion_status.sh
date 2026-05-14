#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  RUN_DIR="$(find artifacts/kinematic_phase1/workspace_expansion -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
else
  RUN_DIR="artifacts/kinematic_phase1/workspace_expansion/$RUN_ID"
fi

if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "FAIL: no workspace expansion run found."
  exit 1
fi

echo "=============================================================================="
echo "WORKSPACE EXPANSION STATUS"
echo "=============================================================================="
echo "run_dir: $RUN_DIR"
if [[ -f "$RUN_DIR/run.pid" ]]; then
  PID="$(cat "$RUN_DIR/run.pid")"
  if [[ -f "$RUN_DIR/run.tmux" ]] && tmux has-session -t "$(cat "$RUN_DIR/run.tmux")" >/dev/null 2>&1; then
    echo "process: RUNNING tmux=$(cat "$RUN_DIR/run.tmux") pid=$PID"
  elif ps -p "$PID" >/dev/null 2>&1; then
    echo "process: RUNNING pid=$PID"
  else
    echo "process: not running pid=$PID"
  fi
fi
if [[ -f "$RUN_DIR/best_checkpoint/model_best_by_gate.zip" ]]; then
  echo "best checkpoint: $RUN_DIR/best_checkpoint/model_best_by_gate.zip"
else
  echo "best checkpoint: not saved yet"
fi
if [[ -f "$RUN_DIR/latest_checkpoint/model_latest.zip" ]]; then
  echo "latest checkpoint: $RUN_DIR/latest_checkpoint/model_latest.zip"
fi

python3 - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
def load(p):
    return json.loads(p.read_text()) if p.exists() else None
selection = load(run / "best_model_selection_summary.json")
if selection:
    print("best gated score:", selection.get("score"))
    print("highest passed stage:", selection.get("highest_passed_stage"))
    print("retention ok:", selection.get("retention_ok"))
metrics = load(run / "stage_metrics.json")
if metrics is None:
    eval_dirs = sorted((run / "gate_evals").glob("eval_step_*")) if (run / "gate_evals").exists() else []
    if eval_dirs:
        metrics = load(eval_dirs[-1] / "stage_metrics.json")
if metrics:
    print("\nStage summary:")
    for key in sorted(metrics, key=lambda x: int(x)):
        row = metrics[key]
        print(
            f"  stage {key}: success={row.get('success_rate', 0):.3f} "
            f"ready={row.get('finisher_ready_hit_rate', 0):.3f} "
            f"pos={row.get('mean_final_position_error', 0):.5f} "
            f"ori={row.get('mean_final_orientation_error', 0):.5f}"
        )
failure = load(run / "workspace_failure_report.json")
if failure is None:
    eval_dirs = sorted((run / "gate_evals").glob("eval_step_*")) if (run / "gate_evals").exists() else []
    if eval_dirs:
        failure = load(eval_dirs[-1] / "workspace_failure_report.json")
if failure:
    print("\nFailure reasons:")
    for stage, counts in failure.get("stage_failure_reason_counts", {}).items():
        print(f"  stage {stage}: {counts}")
hist = run / "eval_history.jsonl"
if hist.exists():
    lines = [l for l in hist.read_text().splitlines() if l.strip()]
    print(f"\neval history records: {len(lines)}")
    for line in lines[-3:]:
        item = json.loads(line)
        print("  ", item)
PY

echo
echo "Log tail:"
if [[ -f "$RUN_DIR/run.log" ]]; then
  tail -40 "$RUN_DIR/run.log"
else
  echo "WARN: no run.log yet"
fi
