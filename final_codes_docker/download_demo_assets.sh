#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${ASSET_CHECK_DIR:-$REPO_ROOT/final_codes_docker/asset_check}"
CHECK_ONLY="${CHECK_ONLY:-0}"
mkdir -p "$OUT_DIR"

python3 - <<'PY'
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    import yaml
except Exception as exc:
    print(f"ERROR: PyYAML is required for asset checking: {exc}", file=sys.stderr)
    sys.exit(2)

repo = Path.cwd()
manifest_path = repo / "final_codes_docker" / "model_manifest.yaml"
out_dir = Path(os.environ.get("ASSET_CHECK_DIR", repo / "final_codes_docker" / "asset_check"))
check_only = os.environ.get("CHECK_ONLY", "0") == "1"

manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
resolved: list[str] = []
missing: list[str] = []
downloaded: list[str] = []


def download_with_gdown(file_id: str, dest: Path) -> bool:
    cmd = [sys.executable, "-m", "gdown", "--id", file_id, "-O", str(dest)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(result.stdout)
    return result.returncode == 0 and dest.exists()


def download_url(url: str, dest: Path) -> bool:
    if "drive.google.com" in url:
        cmd = [sys.executable, "-m", "gdown", url, "-O", str(dest)]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        print(result.stdout)
        return result.returncode == 0 and dest.exists()
    urllib.request.urlretrieve(url, dest)
    return dest.exists()


for item in manifest.get("models", []):
    name = item.get("logical_name", "unnamed")
    rel = item.get("expected_local_path", "")
    if not rel:
        continue
    dest = repo / rel
    if dest.exists():
        resolved.append(f"{name}: {rel}")
        continue
    if check_only:
        missing.append(f"{name}: {rel} (check-only; not downloaded)")
        continue
    url = str(item.get("download_url") or "").strip()
    gid = str(item.get("google_drive_file_id") or "").strip()
    if url or gid:
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = download_with_gdown(gid, dest) if gid else download_url(url, dest)
        if ok:
            resolved.append(f"{name}: {rel}")
            downloaded.append(f"{name}: {rel}")
        else:
            missing.append(f"{name}: {rel} (download failed)")
    else:
        missing.append(
            f"{name}: {rel} (missing; no URL in model_manifest.yaml)"
        )

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "resolved_assets.txt").write_text("\n".join(resolved) + ("\n" if resolved else ""), encoding="utf-8")
(out_dir / "missing_assets.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
(out_dir / "assets_check_summary.txt").write_text(
    "\n".join(
        [
            f"resolved_count: {len(resolved)}",
            f"missing_count: {len(missing)}",
            f"downloaded_count: {len(downloaded)}",
            f"check_only: {check_only}",
            "",
            "resolved:",
            *resolved,
            "",
            "missing:",
            *missing,
        ]
    )
    + "\n",
    encoding="utf-8",
)

print("ASSET CHECK SUMMARY")
print(f"resolved: {len(resolved)}")
print(f"missing: {len(missing)}")
print(f"outputs: {out_dir}")
if missing:
    print("\nMissing assets:")
    for line in missing:
        print(f"  - {line}")
    print("\nPlace missing models at the listed paths, or add URLs to final_codes_docker/model_manifest.yaml.")
PY

