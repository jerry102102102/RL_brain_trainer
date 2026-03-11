#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SMOKE_COUNT=10
RANDOM_COUNT=20
SEED=42

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--smoke-count N] [--random-count N] [--seed N] [--slot-map PATH]

Runs V5 WP1 acceptance harness and prints summary counters.
USAGE
}

SLOT_MAP_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-count)
      SMOKE_COUNT="$2"
      shift 2
      ;;
    --random-count)
      RANDOM_COUNT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --slot-map)
      SLOT_MAP_ARG=(--slot-map "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/hrl_ws/src/hrl_trainer${PYTHONPATH:+:$PYTHONPATH}"

SUMMARY_JSON="$($REPO_ROOT/hrl_ws/.venv/bin/python -m hrl_trainer.v5.pipeline \
  --smoke-count "$SMOKE_COUNT" \
  --random-count "$RANDOM_COUNT" \
  --seed "$SEED" \
  "${SLOT_MAP_ARG[@]}")"

echo "$SUMMARY_JSON"

SUMMARY_JSON="$SUMMARY_JSON" $REPO_ROOT/hrl_ws/.venv/bin/python - <<'PY'
import json
import os

summary = json.loads(os.environ["SUMMARY_JSON"])
overall = summary["overall"]
print("summary:")
print(f"  success_count={overall['success_count']}")
print(f"  fail_count={overall['fail_count']}")
print(f"  fail_reason_breakdown={overall['fail_reason_breakdown']}")
PY
