#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-workspace_full_coverage_randomstart_smoke_001}"
CONFIG="${CONFIG:-hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/workspace_full_coverage_randomstart_overnight.yaml}"
ART_ROOT="artifacts/kinematic_phase1/workspace_full_coverage_randomstart"
RUN_DIR="$ART_ROOT/$RUN_ID"
SMOKE_STEPS="${SMOKE_STEPS:-5000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/hrl_ws/.venv/bin/python3" ]]; then
  PYTHON_BIN="$ROOT/hrl_ws/.venv/bin/python3"
fi
export PYTHONPATH="$ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"
SMOKE_CONFIG="$RUN_DIR/smoke_config.yaml"

"$PYTHON_BIN" - "$CONFIG" "$SMOKE_CONFIG" "$SMOKE_STEPS" <<'PY'
import sys, yaml
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src)) or {}
cfg["base_config"] = cfg.get("base_config", "workspace_expansion_1h_extend.yaml")
cfg.setdefault("training", {}).update({"n_envs": 2, "vec_env": "dummy", "device": "cpu", "checkpoint_freq": 1000})
cfg.setdefault("algorithms", {}).setdefault("ppo", {}).update({
    "total_timesteps": steps,
    "n_steps": 64,
    "batch_size": 64,
    "n_epochs": 1,
    "learning_rate": 1.0e-6,
})
cfg.setdefault("workspace_expansion", {}).update({"gate_eval_episodes": 2, "final_eval_episodes": 4, "eval_interval": 1000})
cfg.setdefault("full_workspace_randomstart", {}).update({
    "target_stage_samples_per_stage": 8,
    "target_random_samples": 24,
    "start_stage_samples_per_stage": 6,
    "start_random_samples": 24,
    "pair_count": 96,
    "eval_episodes_per_split": 6,
})
open(dst, "w").write(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "=============================================================================="
echo "FULL WORKSPACE RANDOM-START SMOKE"
echo "=============================================================================="
echo "run_id: $RUN_ID"
echo "config: $SMOKE_CONFIG"
echo "artifact: $RUN_DIR"
echo "=============================================================================="

echo "[1/4] Python import check"
"$PYTHON_BIN" -m py_compile \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/workspace/workspace_target_map.py \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/workspace/workspace_start_state_map.py \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/workspace/start_target_pair_sampler.py \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/workspace/adaptive_frontier_sampler.py \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/eval/eval_full_workspace_coverage.py \
  hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/train_workspace_expansion.py

echo "[2/4] Generate maps"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.workspace.workspace_target_map \
  --config "$SMOKE_CONFIG" \
  --output-dir "$RUN_DIR/maps" \
  --stage-samples-per-stage 8 \
  --random-samples 24
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.workspace.workspace_start_state_map \
  --config "$SMOKE_CONFIG" \
  --output-dir "$RUN_DIR/maps" \
  --stage-samples-per-stage 6 \
  --random-samples 24
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.workspace.start_target_pair_sampler \
  --start-map "$RUN_DIR/maps/start_state_map.jsonl" \
  --target-map "$RUN_DIR/maps/target_map.jsonl" \
  --output-dir "$RUN_DIR/maps" \
  --pair-count 96

echo "[3/4] Tiny PPO reset/training smoke"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.train_workspace_expansion \
  --config "$SMOKE_CONFIG" \
  --run-id "$RUN_ID" \
  --artifact-root "$RUN_DIR" \
  --total-timesteps "$SMOKE_STEPS" \
  --no-gate-callback 2>&1 | tee "$RUN_DIR/run.log"

echo "[4/4] Random-start coverage eval smoke"
"$PYTHON_BIN" -m hrl_trainer.kinematic_phase1.eval.eval_full_workspace_coverage \
  --approach-checkpoint "$RUN_DIR/latest_checkpoint/model_latest.zip" \
  --approach-config "$SMOKE_CONFIG" \
  --finisher-checkpoint artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip \
  --finisher-config hrl_ws/src/hrl_trainer/hrl_trainer/kinematic_phase1/configs/dock_workspace_handoff_noop_ft_12env.yaml \
  --artifact-root "$RUN_DIR/smoke_full_workspace_eval" \
  --episodes-per-split 4 \
  --stage-samples-per-stage 6 \
  --random-target-samples 18 \
  --random-start-samples 18 \
  --pair-count 72 \
  --skip-home-stage-eval

echo "SMOKE PASS"
echo "artifact: $RUN_DIR"
