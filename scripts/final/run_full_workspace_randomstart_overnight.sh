#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-workspace_full_coverage_randomstart_overnight_001}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-workspace_full_coverage_randomstart_smoke_001}"
CONFIG="${CONFIG:-hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/workspace_full_coverage_randomstart_overnight.yaml}"
ART_ROOT="artifacts/kinematic_phase1/workspace_full_coverage_randomstart"
RUN_DIR="$ART_ROOT/$RUN_ID"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-8000000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/hrl_ws/.venv/bin/python3" ]]; then
  PYTHON_BIN="$ROOT/hrl_ws/.venv/bin/python3"
fi
export PYTHONPATH="$ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"

echo "=============================================================================="
echo "FULL WORKSPACE RANDOM-START OVERNIGHT TRAINING"
echo "=============================================================================="
echo "run_id: $RUN_ID"
echo "config: $CONFIG"
echo "artifact: $RUN_DIR"
echo "total_timesteps: $TOTAL_TIMESTEPS"
echo "=============================================================================="

echo "[1/3] Smoke preflight"
RUN_ID="$SMOKE_RUN_ID" CONFIG="$CONFIG" SMOKE_STEPS="${SMOKE_STEPS:-3000}" bash scripts/final/run_full_workspace_randomstart_smoke.sh

echo "[2/3] Launching long PPO run in background"
cat > "$RUN_DIR/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$ROOT/hrl_ws/src/hrl_trainer:\${PYTHONPATH:-}"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.train_workspace_expansion \\
  --config "$CONFIG" \\
  --run-id "$RUN_ID" \\
  --artifact-root "$RUN_DIR" \\
  --total-timesteps "$TOTAL_TIMESTEPS"
EVAL_CKPT="$RUN_DIR/best_checkpoint/model_best_by_gate.zip"
if [[ ! -f "\$EVAL_CKPT" ]]; then
  EVAL_CKPT="$RUN_DIR/latest_checkpoint/model_latest.zip"
fi
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.eval.eval_full_workspace_coverage \\
  --approach-checkpoint "\$EVAL_CKPT" \\
  --approach-config "$CONFIG" \\
  --finisher-checkpoint artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip \\
  --finisher-config hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml \\
  --artifact-root "$RUN_DIR/final_full_workspace_eval" \\
  --episodes-per-split 96 \\
  --stage-samples-per-stage 96 \\
  --random-target-samples 384 \\
  --random-start-samples 384 \\
  --pair-count 2048 || true
python3 scripts/final/plot_workspace_expansion.py --run-dir "$RUN_DIR" || true
EOF
chmod +x "$RUN_DIR/run_command.sh"

if command -v tmux >/dev/null 2>&1; then
  SESSION="ws_full_random_${RUN_ID}"
  tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$RUN_DIR/run_command.sh' > '$RUN_DIR/run.log' 2>&1"
  echo "$SESSION" > "$RUN_DIR/run.tmux"
  PID="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1)"
  echo "$PID" > "$RUN_DIR/run.pid"
else
  setsid "$RUN_DIR/run_command.sh" > "$RUN_DIR/run.log" 2>&1 &
  PID=$!
  echo "$PID" > "$RUN_DIR/run.pid"
fi

echo "[3/3] Started"
echo "pid: $PID"
echo "log: $RUN_DIR/run.log"
if [[ -f "$RUN_DIR/run.tmux" ]]; then
  echo "tmux: tmux attach -t $(cat "$RUN_DIR/run.tmux")"
fi
echo "status: bash scripts/final/check_full_workspace_randomstart_status.sh $RUN_ID"
