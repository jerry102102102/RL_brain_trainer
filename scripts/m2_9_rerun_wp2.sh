#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HRL_ROOT="$REPO_ROOT/hrl_ws/src/hrl_trainer"
OUT_DIR="$REPO_ROOT/artifacts/reports/v5"
PYTHON_BIN="$REPO_ROOT/hrl_ws/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p "$OUT_DIR"

cd "$HRL_ROOT"

"$PYTHON_BIN" -m unittest -q \
  tests.test_v5_m2_7_training_loop_integration \
  tests.test_v5_benchmark_rule_l2_v0 \
  tests.test_v5_eval_harness \
  tests.test_v5_m2_8_formal_comparison

"$PYTHON_BIN" -m hrl_trainer.v5.eval_harness \
  --policy rule_l2_v0 \
  --episodes 8 \
  --seed 42 \
  --strict-policy \
  --output "$OUT_DIR/v5_eval_rule_l2_v0_seed42_ep8.json"

"$PYTHON_BIN" -m hrl_trainer.v5.eval_harness \
  --policy rl_l2 \
  --episodes 4 \
  --seed 11 \
  --strict-policy \
  --output "$OUT_DIR/v5_eval_rl_l2_seed11_ep4.json"

"$PYTHON_BIN" -m hrl_trainer.v5.eval_harness \
  --policy rl_l2 \
  --episodes 4 \
  --seed 13 \
  --strict-policy \
  --output "$OUT_DIR/v5_eval_rl_l2_seed13_ep4.json"

"$PYTHON_BIN" -m hrl_trainer.v5.benchmark_m2_8_formal_comparison \
  --episodes 8 \
  --seed 42 \
  --output "$OUT_DIR/m2_8_formal_comparison_summary_seed42_ep8.json"

printf "WP2 rerun complete. Outputs:\n"
ls -1 "$OUT_DIR"/v5_eval_rule_l2_v0_seed42_ep8.json \
      "$OUT_DIR"/v5_eval_rl_l2_seed11_ep4.json \
      "$OUT_DIR"/v5_eval_rl_l2_seed13_ep4.json \
      "$OUT_DIR"/m2_8_formal_comparison_summary_seed42_ep8.json
