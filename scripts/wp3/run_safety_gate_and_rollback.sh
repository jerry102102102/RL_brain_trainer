#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-8}"
SEEDS="${SEEDS:-42,43}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"; shift 2;;
    --seeds)
      SEEDS="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HRL_ROOT="$REPO_ROOT/hrl_ws/src/hrl_trainer"
PYTHON_BIN="$REPO_ROOT/hrl_ws/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

TS="$(date +%F_%H%M%S)"
OUT_DIR="$REPO_ROOT/artifacts/reports/wp3/$TS"
OUT_JSON="$OUT_DIR/ws3_rollback_gate.json"
mkdir -p "$OUT_DIR"

cd "$HRL_ROOT"
PYTHONPATH="$HRL_ROOT" "$PYTHON_BIN" -m hrl_trainer.v5.wp3_gates ws3 \
  --episodes "$EPISODES" \
  --seeds "$SEEDS" \
  --output "$OUT_JSON"

echo "WS3 rollback evidence: $OUT_JSON"
