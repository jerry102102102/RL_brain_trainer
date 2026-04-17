#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-mock}"
POLICY="${POLICY:-rule_l2_v0}"
SEED="${SEED:-42}"
NOTES="${NOTES:-WP3 WS1 dry-run evidence; no physical robot motion executed.}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2;;
    --policy)
      POLICY="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --notes)
      NOTES="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ "$MODE" != "mock" && "$MODE" != "dryrun" && "$MODE" != "real" ]]; then
  echo "Unsupported --mode '$MODE' (allowed: mock|dryrun|real)" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%F_%H%M%S)"
OUT_DIR="$REPO_ROOT/artifacts/wp3/hil_dryrun/$TS"
OUT_JSON="$OUT_DIR/hil_runtime_evidence.json"
OUT_LOG="$OUT_DIR/hil_runtime_evidence.log"
SCHEMA_PATH="$REPO_ROOT/docs/wp3/hil_runtime_evidence.schema.json"

mkdir -p "$OUT_DIR"

health_ok=true
topic_ok=true
bridge_ok=true
health_detail="check passed"
topic_detail="check passed"
bridge_detail="check passed"
runtime_source="simulated_hil_dryrun"
pass_value=true

if [[ "$MODE" == "mock" || "$MODE" == "dryrun" ]]; then
  health_detail="$MODE health check passed (synthetic)"
  topic_detail="$MODE topic check passed (synthetic)"
  bridge_detail="$MODE bridge check passed (synthetic)"
  runtime_source="simulated_hil_dryrun"
else
  runtime_source="real_robot_hil"

  if ! command -v ros2 >/dev/null 2>&1; then
    health_ok=false
    topic_ok=false
    health_detail="ros2 CLI not found in PATH"
    topic_detail="ros2 CLI not found in PATH"
  else
    if ros2 node list >/tmp/wp3_ros2_nodes.$$ 2>/dev/null; then
      if [[ ! -s /tmp/wp3_ros2_nodes.$$ ]]; then
        health_ok=false
        health_detail="ros2 reachable but no nodes detected"
      else
        health_detail="ros2 node list returned active nodes"
      fi
    else
      health_ok=false
      health_detail="failed to query ros2 node list"
    fi

    if ros2 topic list >/tmp/wp3_ros2_topics.$$ 2>/dev/null; then
      if grep -qE '^/tf$|^/clock$' /tmp/wp3_ros2_topics.$$; then
        topic_detail="required topic probe (/tf or /clock) found"
      else
        topic_ok=false
        topic_detail="ros2 topic list succeeded but missing /tf and /clock"
      fi
    else
      topic_ok=false
      topic_detail="failed to query ros2 topic list"
    fi
  fi

  if pgrep -fa 'parameter_bridge|l3_runtime_bridge' >/tmp/wp3_bridge.$$ 2>/dev/null; then
    bridge_detail="bridge process detected"
  else
    bridge_ok=false
    bridge_detail="no parameter_bridge/l3_runtime_bridge process found"
  fi

  rm -f /tmp/wp3_ros2_nodes.$$ /tmp/wp3_ros2_topics.$$ /tmp/wp3_bridge.$$ || true
fi

if [[ "$health_ok" != "true" || "$topic_ok" != "true" || "$bridge_ok" != "true" ]]; then
  pass_value=false
fi

{
  echo "[wp3][hil-dryrun] start ts=$TS mode=$MODE policy=$POLICY seed=$SEED"
  echo "[wp3][hil-dryrun] runtime_source=$runtime_source"
  echo "[wp3][hil-dryrun] checks: health=$health_ok topic=$topic_ok bridge=$bridge_ok"
  echo "[wp3][hil-dryrun] details: health='$health_detail' topic='$topic_detail' bridge='$bridge_detail'"
  echo "[wp3][hil-dryrun] schema=$SCHEMA_PATH"
} | tee "$OUT_LOG"

python3 - <<PY > "$OUT_JSON"
import json
from datetime import datetime, timezone

def to_bool(v: str) -> bool:
    return v.strip().lower() == "true"

payload = {
  "timestamp": datetime.now(timezone.utc).isoformat(),
  "mode": "$MODE",
  "runtime_source": "$runtime_source",
  "policy": "$POLICY",
  "seed": int("$SEED"),
  "checks": {
    "health": {"ok": to_bool("$health_ok"), "detail": "$health_detail"},
    "topic": {"ok": to_bool("$topic_ok"), "detail": "$topic_detail"},
    "bridge": {"ok": to_bool("$bridge_ok"), "detail": "$bridge_detail"},
  },
  "pass": to_bool("$pass_value"),
  "notes": "$NOTES",
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

{
  echo "[wp3][hil-dryrun] wrote json=$OUT_JSON"
  echo "[wp3][hil-dryrun] wrote log=$OUT_LOG"
} | tee -a "$OUT_LOG"

echo "$OUT_DIR"

if [[ "$MODE" == "real" && "$pass_value" != "true" ]]; then
  exit 3
fi
