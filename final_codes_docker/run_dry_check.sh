#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/hrl_ws/.venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hrl_ws/.venv/bin/activate"
  set -u
fi

export PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"

pass() { echo "PASS $*"; }
warn() { echo "WARN $*"; }
fail() { echo "FAIL $*"; exit 1; }

command -v python3 >/dev/null && pass "python3" || fail "python3 missing"
python3 - <<'PY'
import importlib
for name in ["numpy", "yaml", "gymnasium", "stable_baselines3"]:
    importlib.import_module(name)
print("python dependencies import ok")
import hrl_trainer
print("hrl_trainer import ok")
PY

for path in \
  final_codes_docker/model_manifest.yaml \
  final_codes_docker/demo_manifest.yaml \
  final_codes_docker/run_local_test_demo.sh \
  final_codes_docker/run_full_route_demo.sh \
  scripts/final/run_final_local_test_demo.sh \
  scripts/final/run_final_full_route_demo.sh
do
  [[ -f "$path" ]] && pass "$path" || fail "$path missing"
done

CHECK_ONLY=1 bash final_codes_docker/download_demo_assets.sh || warn "asset check reported missing assets"

if command -v ros2 >/dev/null 2>&1; then
  pass "ros2 available"
else
  warn "ros2 not available; Docker headless mode can still run numeric checks"
fi

echo "FINAL CODES DOCKER DRY CHECK FINISHED"
