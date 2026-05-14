#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f "$PWD/hrl_ws/.venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$PWD/hrl_ws/.venv/bin/activate"
  set -u
fi

export PYTHONPATH="$PWD/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"

FAILS=0
WARNS=0

pass() { echo "PASS $*"; }
warn() { echo "WARN $*"; WARNS=$((WARNS + 1)); }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }

for cmd in python3 bash; do
  command -v "$cmd" >/dev/null 2>&1 && pass "command:$cmd" || fail "command:$cmd missing"
done

if command -v docker >/dev/null 2>&1; then
  pass "command:docker $(docker --version | head -1)"
else
  warn "docker missing; native scripts still work"
fi

if docker compose version >/dev/null 2>&1; then
  pass "docker compose $(docker compose version | head -1)"
else
  warn "docker compose unavailable"
fi

python3 - <<'PY' && pass "python imports" || fail "python imports failed"
import hrl_trainer
import yaml
print("hrl_trainer:", hrl_trainer.__file__)
PY

for path in \
  final_codes_docker/README_FINAL_CODES_DOCKER.md \
  final_codes_docker/Dockerfile.demo \
  final_codes_docker/docker-compose.demo.yaml \
  final_codes_docker/model_manifest.yaml \
  final_codes_docker/demo_manifest.yaml \
  final_codes_docker/download_demo_assets.sh \
  final_codes_docker/run_local_test_demo.sh \
  final_codes_docker/run_full_route_demo.sh \
  final_codes_docker/run_dry_check.sh \
  final_codes_docker/RECORDING_GUIDE.md \
  scripts/final/run_final_local_test_demo.sh \
  scripts/final/run_final_full_route_demo.sh
do
  [[ -f "$path" ]] && pass "$path" || fail "$path missing"
done

CHECK_ONLY=1 bash final_codes_docker/download_demo_assets.sh || warn "asset checker reported missing models"

if command -v ros2 >/dev/null 2>&1; then
  pass "ros2 available"
else
  warn "ros2 unavailable; Gazebo GUI demo requires native ROS2 setup"
fi

if command -v gz >/dev/null 2>&1; then
  pass "gz available"
else
  warn "gz unavailable; headless Docker path does not require Gazebo GUI"
fi

echo "=============================================================================="
if [[ "$FAILS" -gt 0 ]]; then
  echo "FINAL DEMO READY CHECK: FAIL ($FAILS fail, $WARNS warn)"
  exit 1
fi
echo "FINAL DEMO READY CHECK: PASS ($WARNS warn)"
