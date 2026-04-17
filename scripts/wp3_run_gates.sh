#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-11,13,17}"
EPISODES="${EPISODES:-4}"
ROLLBACK_EPISODES="${ROLLBACK_EPISODES:-8}"
ROLLBACK_SEEDS="${ROLLBACK_SEEDS:-42,43}"
WITH_HIL_DRYRUN="${WITH_HIL_DRYRUN:-0}"
HIL_MODE="${HIL_MODE:-mock}"
HIL_POLICY="${HIL_POLICY:-rule_l2_v0}"
HIL_SEED="${HIL_SEED:-42}"
HIL_EVIDENCE_MODE="${HIL_EVIDENCE_MODE:-any}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS="$2"; shift 2;;
    --episodes)
      EPISODES="$2"; shift 2;;
    --rollback-episodes)
      ROLLBACK_EPISODES="$2"; shift 2;;
    --rollback-seeds)
      ROLLBACK_SEEDS="$2"; shift 2;;
    --with-hil-dryrun)
      WITH_HIL_DRYRUN=1; shift 1;;
    --hil-mode)
      HIL_MODE="$2"; shift 2;;
    --hil-policy)
      HIL_POLICY="$2"; shift 2;;
    --hil-seed)
      HIL_SEED="$2"; shift 2;;
    --hil-evidence-mode)
      HIL_EVIDENCE_MODE="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%F_%H%M%S)"
RUN_DIR="$REPO_ROOT/artifacts/reports/wp3/$TS"
mkdir -p "$RUN_DIR"

if [[ "$WITH_HIL_DRYRUN" == "1" ]]; then
  "$REPO_ROOT/scripts/wp3/run_hil_dryrun_and_capture.sh" \
    --mode "$HIL_MODE" \
    --policy "$HIL_POLICY" \
    --seed "$HIL_SEED" | tee "$RUN_DIR/ws1_hil_dryrun_stdout.log"
fi

# WS1
set +e
WP3_HIL_EVIDENCE_MODE="$HIL_EVIDENCE_MODE" \
"$REPO_ROOT/scripts/wp3/run_hil_gate.sh" | tee "$RUN_DIR/ws1_stdout.log"
WS1_RC=$?
set -e

# WS2
"$REPO_ROOT/scripts/wp3/run_seed_episode_matrix.sh" --seeds "$SEEDS" --episodes "$EPISODES" | tee "$RUN_DIR/ws2_stdout.log"

# WS3
set +e
"$REPO_ROOT/scripts/wp3/run_safety_gate_and_rollback.sh" --episodes "$ROLLBACK_EPISODES" --seeds "$ROLLBACK_SEEDS" | tee "$RUN_DIR/ws3_stdout.log"
WS3_RC=$?
set -e

STATUS_JSON="$RUN_DIR/wp3_gate_status.json"
python3 - <<PY > "$STATUS_JSON"
import json
from pathlib import Path
run_dir = Path("$RUN_DIR")
status = {
  "ws1_exit_code": $WS1_RC,
  "ws2_exit_code": 0,
  "ws3_exit_code": $WS3_RC,
  "overall_pass": ($WS1_RC == 0 and $WS3_RC == 0),
  "with_hil_dryrun": bool($WITH_HIL_DRYRUN),
  "note": "WS1 requires valid HIL evidence JSON + checks + pass=true under artifacts/wp3/hil_dryrun."
}
print(json.dumps(status, indent=2, sort_keys=True))
PY

cat "$STATUS_JSON"
echo "WP3 gate run bundle: $RUN_DIR"

if [[ $WS1_RC -ne 0 || $WS3_RC -ne 0 ]]; then
  exit 2
fi
