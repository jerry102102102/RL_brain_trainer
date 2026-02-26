#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_SCENE_REPO="/home/jerry/.openclaw/workspace/repos/personal/ENPM662_Group4_FinalProject"
DEFAULT_LINK_PATH="$REPO_ROOT/external/kitchen_scene"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--scene-repo PATH] [--link-path PATH] [--validate-only] [--dry-run]

Purpose:
  Bridge V5 WP0 runtime to the ENPM662 kitchen scene repo via symlink (no asset duplication).

Options:
  --scene-repo PATH   Source kitchen scene repo path.
                      Default: $DEFAULT_SCENE_REPO
  --link-path PATH    Symlink path to create in V5 repo.
                      Default: $DEFAULT_LINK_PATH
  --validate-only     Validate inputs/bridge status without modifying files.
  --dry-run           Print planned actions only.
  -h, --help          Show this help.
USAGE
}

SCENE_REPO="$DEFAULT_SCENE_REPO"
LINK_PATH="$DEFAULT_LINK_PATH"
VALIDATE_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-repo)
      SCENE_REPO="$2"
      shift 2
      ;;
    --link-path)
      LINK_PATH="$2"
      shift 2
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
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

if [[ ! -d "$SCENE_REPO" ]]; then
  cat >&2 <<ERR
ERROR: scene repo not found: $SCENE_REPO
Fix:
  1) Clone/checkout ENPM662 scene repo at that path, or
  2) Re-run with --scene-repo <existing-path>
ERR
  exit 2
fi

if [[ ! -d "$SCENE_REPO/.git" ]]; then
  echo "WARNING: $SCENE_REPO exists but does not look like a git repo (.git missing)." >&2
fi

if [[ -L "$LINK_PATH" ]]; then
  CURRENT_TARGET="$(readlink "$LINK_PATH")"
  if [[ "$CURRENT_TARGET" == "$SCENE_REPO" ]]; then
    echo "OK: bridge already configured: $LINK_PATH -> $CURRENT_TARGET"
    exit 0
  fi
fi

if [[ -e "$LINK_PATH" && ! -L "$LINK_PATH" ]]; then
  echo "ERROR: link path exists and is not a symlink: $LINK_PATH" >&2
  echo "Remove or move it, then re-run bridge setup." >&2
  exit 3
fi

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
  if [[ -L "$LINK_PATH" ]]; then
    echo "OK: validate-only passed (existing symlink): $LINK_PATH -> $(readlink "$LINK_PATH")"
  else
    echo "OK: validate-only passed (source repo exists, bridge not yet created)."
    echo "Run without --validate-only to create: $LINK_PATH -> $SCENE_REPO"
  fi
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would create bridge symlink"
  echo "  from: $SCENE_REPO"
  echo "  to:   $LINK_PATH"
  exit 0
fi

mkdir -p "$(dirname "$LINK_PATH")"
rm -f "$LINK_PATH"
ln -s "$SCENE_REPO" "$LINK_PATH"

echo "OK: bridge created"
echo "  $LINK_PATH -> $(readlink "$LINK_PATH")"
