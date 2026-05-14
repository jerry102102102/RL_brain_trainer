#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-workspace_expand_stage6to9_ppo_big_001}"
RUN_DIR="artifacts/kinematic_phase1/workspace_expansion/$RUN_ID"
SESSION="ws_expand_${RUN_ID}"

if [[ ! -x "$RUN_DIR/run_command.sh" ]]; then
  echo "FAIL: missing $RUN_DIR/run_command.sh"
  exit 1
fi

export TERM="${TERM:-xterm-256color}"
tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$RUN_DIR/run_command.sh' > '$RUN_DIR/run.log' 2>&1"
tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -1 > "$RUN_DIR/run.pid"
echo "$SESSION" > "$RUN_DIR/run.tmux"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
