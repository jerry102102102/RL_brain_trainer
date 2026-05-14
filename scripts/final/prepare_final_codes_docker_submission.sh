#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT_ZIP="${OUT_ZIP:-report/final_codes_docker_submission.zip}"
INCLUDE_MODELS="${INCLUDE_MODELS:-1}"

mkdir -p report

python3 - <<'PY'
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

repo = Path.cwd()
zip_path = repo / os.environ.get("OUT_ZIP", "report/final_codes_docker_submission.zip")
include_models = os.environ.get("INCLUDE_MODELS", "1") == "1"
staging = repo / "report" / "_final_codes_docker_submission_staging"

if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

def copy_path(rel: str) -> None:
    src = repo / rel
    dst = staging / rel
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "asset_check"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

for rel in [
    "README.md",
    "REPORT_EXTRA_DETAILS.md",
    "final_codes_docker",
    "scripts/final/run_final_local_test_demo.sh",
    "scripts/final/run_final_full_route_demo.sh",
    "scripts/final/check_final_demo_ready.sh",
    "scripts/final/prepare_final_codes_docker_submission.sh",
    "scripts/final/run_live_gz_screen_recording_demo.sh",
    "scripts/final/run_live_gz_vlm_demo.sh",
    "scripts/final/generate_final_figures.py",
    "scripts/final/run_demo_02_kinematic_skill.sh",
    "scripts/final/run_demo_03_route_curriculum.sh",
    "config/rviz/phase3a_demo.rviz",
    "docs/CURRENT_IMPLEMENTATION.md",
    "docs/README.md",
    "report/FINAL_PROJECT_SUMMARY.md",
    "report/OFFICIAL_ARTIFACTS.md",
    "report/DEMO_VIDEO_SCRIPT.md",
    "report/DEMO_RECORDING_COMMANDS.md",
    "hrl_ws/src/hrl_trainer",
]:
    copy_path(rel)

if include_models and yaml is not None:
    manifest = yaml.safe_load((repo / "final_codes_docker" / "model_manifest.yaml").read_text())
    for item in manifest.get("models", []):
        rel = item.get("expected_local_path")
        if rel and (repo / rel).exists():
            copy_path(rel)

zip_path.parent.mkdir(parents=True, exist_ok=True)
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in staging.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(staging))

shutil.rmtree(staging)
print(f"created: {zip_path}")
print(f"include_models: {include_models}")
PY

echo "FINAL CODES DOCKER SUBMISSION PACKAGE READY"
echo "zip: $OUT_ZIP"

