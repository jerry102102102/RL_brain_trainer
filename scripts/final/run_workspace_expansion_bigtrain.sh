#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/workspace_expansion_bigtrain.yaml}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-workspace_expand_stage6to9_ppo_smoke_001}"
BIG_RUN_ID="${RUN_ID:-workspace_expand_stage6to9_ppo_big_001}"
SMOKE_STEPS="${SMOKE_STEPS:-5000}"
BIG_STEPS="${TOTAL_TIMESTEPS:-5000000}"
ART_ROOT="artifacts/kinematic_phase1/workspace_expansion"
SMOKE_DIR="$ART_ROOT/$SMOKE_RUN_ID"
BIG_DIR="$ART_ROOT/$BIG_RUN_ID"

export PYTHONPATH="$ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/hrl_ws/.venv/bin/python3" ]]; then
  PYTHON_BIN="$ROOT/hrl_ws/.venv/bin/python3"
fi

mkdir -p "$SMOKE_DIR" "$BIG_DIR"
SMOKE_CONFIG="$SMOKE_DIR/smoke_config.yaml"
"$PYTHON_BIN" - "$CONFIG" "$SMOKE_CONFIG" "$SMOKE_STEPS" <<'PY'
import sys, yaml
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("training", {})
cfg["training"].update({"n_envs": 2, "vec_env": "dummy", "device": "cpu", "checkpoint_freq": 1000})
cfg.setdefault("algorithms", {}).setdefault("ppo", {})
cfg["algorithms"]["ppo"].update({"total_timesteps": steps, "n_steps": 64, "batch_size": 64, "n_epochs": 1})
cfg.setdefault("workspace_expansion", {})
cfg["workspace_expansion"].update({"gate_eval_episodes": 2, "final_eval_episodes": 4, "eval_interval": 1000})
open(dst, "w").write(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "=============================================================================="
echo "WORKSPACE EXPANSION CURRICULUM BIG TRAINING"
echo "=============================================================================="
echo "config: $CONFIG"
echo "smoke run: $SMOKE_RUN_ID ($SMOKE_STEPS steps)"
echo "big run:   $BIG_RUN_ID ($BIG_STEPS steps)"
echo "artifact:  $BIG_DIR"
echo "=============================================================================="

echo "[1/3] Smoke test"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.train_workspace_expansion \
  --config "$SMOKE_CONFIG" \
  --run-id "$SMOKE_RUN_ID" \
  --artifact-root "$SMOKE_DIR" \
  --total-timesteps "$SMOKE_STEPS" \
  --no-gate-callback 2>&1 | tee "$SMOKE_DIR/run.log"

echo "[2/3] Smoke eval stages 0,5,6"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.eval.eval_workspace_expansion \
  --approach-checkpoint "$SMOKE_DIR/latest_checkpoint/model_latest.zip" \
  --approach-config "$SMOKE_CONFIG" \
  --finisher-checkpoint artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip \
  --finisher-config hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml \
  --artifact-root "$SMOKE_DIR/smoke_eval" \
  --episodes 4 \
  --stages 0,5,6

echo "[3/3] Launching long training in background"
cat > "$BIG_DIR/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$ROOT/hrl_ws/src/hrl_trainer:\${PYTHONPATH:-}"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.train_workspace_expansion \\
  --config "$CONFIG" \\
  --run-id "$BIG_RUN_ID" \\
  --artifact-root "$BIG_DIR" \\
  --total-timesteps "$BIG_STEPS"
"$PYTHON_BIN" scripts/final/plot_workspace_expansion.py --run-dir "$BIG_DIR" || true
EOF
chmod +x "$BIG_DIR/run_command.sh"
if command -v tmux >/dev/null 2>&1; then
  SESSION="ws_expand_${BIG_RUN_ID}"
  tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
  tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$BIG_DIR/run_command.sh' > '$BIG_DIR/run.log' 2>&1"
  echo "$SESSION" > "$BIG_DIR/run.tmux"
  PID="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1)"
  echo "$PID" > "$BIG_DIR/run.pid"
else
  setsid "$BIG_DIR/run_command.sh" > "$BIG_DIR/run.log" 2>&1 &
  PID=$!
  echo "$PID" > "$BIG_DIR/run.pid"
fi

echo "Long training started."
echo "pid: $PID"
echo "log: $BIG_DIR/run.log"
if [[ -f "$BIG_DIR/run.tmux" ]]; then
  echo "tmux: tmux attach -t $(cat "$BIG_DIR/run.tmux")"
fi
echo "status: bash scripts/final/check_workspace_expansion_status.sh $BIG_RUN_ID"
