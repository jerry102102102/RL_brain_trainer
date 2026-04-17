#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-11,13,17}"
EPISODES="${EPISODES:-4}"
POLICY="${POLICY:-rl_l2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS="$2"; shift 2;;
    --episodes)
      EPISODES="$2"; shift 2;;
    --policy)
      POLICY="$2"; shift 2;;
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
PER_SEED_DIR="$OUT_DIR/ws2_per_seed"
OUT_JSON="$OUT_DIR/ws2_seed_sweep_summary.json"
mkdir -p "$OUT_DIR"

cd "$HRL_ROOT"
PYTHONPATH="$HRL_ROOT" "$PYTHON_BIN" -m hrl_trainer.v5.wp3_gates ws2 \
  --seeds "$SEEDS" \
  --episodes "$EPISODES" \
  --policy "$POLICY" \
  --per-seed-dir "$PER_SEED_DIR" \
  --output "$OUT_JSON"

echo "WS2 summary: $OUT_JSON"
