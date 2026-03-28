#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EXPECTED_PY="$ROOT_DIR/hrl_ws/.venv/bin/python"

source "$ROOT_DIR/scripts/v5_1/activate_env.sh" >/dev/null

ACTUAL_PY="$(command -v python)"
if [[ "$ACTUAL_PY" != "$EXPECTED_PY" ]]; then
  echo "[FAIL] python path mismatch"
  echo "  expected: $EXPECTED_PY"
  echo "  actual:   $ACTUAL_PY"
  exit 1
fi

python - <<'PY'
import importlib.util
import sys

mods = ("torch", "rclpy")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print(f"[FAIL] missing modules: {', '.join(missing)}")
    raise SystemExit(1)
print("[PASS] python:", sys.executable)
print("[PASS] torch+rclpy import check ok")
PY

if [[ -z "${PYTHONPATH:-}" ]]; then
  echo "[FAIL] PYTHONPATH is empty"
  exit 1
fi

echo "[PASS] PYTHONPATH=$PYTHONPATH"
