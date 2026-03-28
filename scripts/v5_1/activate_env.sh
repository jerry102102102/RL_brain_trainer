#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

set +u
source /opt/ros/jazzy/setup.zsh

EXT_SETUP="$ROOT_DIR/external/kitchen_scene/src/install/setup.zsh"
if [[ -f "$EXT_SETUP" ]]; then
  source "$EXT_SETUP"
fi
set -u

source "$ROOT_DIR/hrl_ws/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR/hrl_ws/src/hrl_trainer:${PYTHONPATH:-}"

echo "[v5_1] env activated"
echo "[v5_1] python=$(command -v python)"
echo "[v5_1] PYTHONPATH=$PYTHONPATH"
