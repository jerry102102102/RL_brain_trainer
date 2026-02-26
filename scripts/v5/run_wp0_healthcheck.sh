#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--live] [--replay-bag PATH] [--output PATH] [--dry-run]

Purpose:
  Run V5 WP0 unified healthcheck with standard config/artifacts paths.

Options:
  --live              Enable live metrics mode.
  --replay-bag PATH   Replay bag path for rosbag replay checks.
  --output PATH       Output report path.
                      Default: $REPO_ROOT/artifacts/wp0/wp0_report.json
  --dry-run           Print resolved command without execution.
  -h, --help          Show this help.
USAGE
}

LIVE=0
REPLAY_BAG=""
OUTPUT="$REPO_ROOT/artifacts/wp0/wp0_report.json"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      LIVE=1
      shift
      ;;
    --replay-bag)
      REPLAY_BAG="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

CONFIG="$REPO_ROOT/hrl_ws/src/hrl_trainer/config/wp0_config.yaml"
ART_DIR="$REPO_ROOT/artifacts/wp0"
mkdir -p "$ART_DIR"

CMD=(python3 -m hrl_trainer.v5.tools.wp0_healthcheck --config "$CONFIG" --artifacts-dir "$ART_DIR" --output "$OUTPUT")
if [[ "$LIVE" -eq 1 ]]; then
  CMD+=(--live)
fi
if [[ -n "$REPLAY_BAG" ]]; then
  CMD+=(--replay-bag "$REPLAY_BAG")
fi

echo "Command: PYTHONPATH=$REPO_ROOT/hrl_ws/src/hrl_trainer ${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: healthcheck skipped"
  exit 0
fi

(
  cd "$REPO_ROOT"
  PYTHONPATH=hrl_ws/src/hrl_trainer "${CMD[@]}"
)
