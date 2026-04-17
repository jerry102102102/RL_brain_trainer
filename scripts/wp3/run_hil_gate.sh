#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HRL_ROOT="$REPO_ROOT/hrl_ws/src/hrl_trainer"
PYTHON_BIN="$REPO_ROOT/hrl_ws/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

TS="$(date +%F_%H%M%S)"
OUT_DIR="$REPO_ROOT/artifacts/reports/wp3/$TS"
OUT_JSON="$OUT_DIR/ws1_runtime_hil_gate.json"
mkdir -p "$OUT_DIR"

cd "$HRL_ROOT"
PYTHONPATH="$HRL_ROOT" "$PYTHON_BIN" -m hrl_trainer.v5.wp3_gates ws1 --output "$OUT_JSON"

echo "WS1 evidence: $OUT_JSON"
