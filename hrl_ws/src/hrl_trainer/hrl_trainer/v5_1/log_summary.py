"""Summarize V5.1 L1/L2/L3 JSONL logs for debugging and health checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS: dict[str, set[str]] = {
    "l1": {"run_id", "episode", "step", "ts", "intent", "stage", "goal_summary"},
    "l2": {"run_id", "episode", "step", "ts", "action_raw", "action_clipped", "delta_q", "policy_status"},
    "l3": {"run_id", "episode", "step", "ts", "q_des", "q_actual", "intervention_type", "reason"},
}


def _iter_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            out.append(payload)
    return out


def _layer_files(logs_root: Path, layer: str) -> list[Path]:
    layer_dir = logs_root / layer
    if not layer_dir.exists():
        return []
    return sorted(p for p in layer_dir.iterdir() if p.is_file() and p.suffix == ".jsonl")


def summarize_logs(logs_root: Path) -> dict[str, Any]:
    logs_root = Path(logs_root)
    summary: dict[str, Any] = {
        "logs_root": str(logs_root),
        "step_count": {"l1": 0, "l2": 0, "l3": 0},
        "intervention_rate": 0.0,
        "action_saturation_rate": 0.0,
        "missing_fields": {"l1": {}, "l2": {}, "l3": {}},
    }

    l2_saturated = 0
    l3_interventions = 0

    for layer in ("l1", "l2", "l3"):
        required = REQUIRED_FIELDS[layer]
        layer_missing: dict[str, int] = {key: 0 for key in required}

        for file_path in _layer_files(logs_root, layer):
            records = _iter_records(file_path)
            summary["step_count"][layer] += len(records)
            for rec in records:
                payload = rec.get("payload", {}) if isinstance(rec, dict) else {}
                for key in required:
                    if key not in payload:
                        layer_missing[key] += 1

                if layer == "l2":
                    status = payload.get("policy_status", {})
                    if isinstance(status, dict) and bool(status.get("saturated", False)):
                        l2_saturated += 1

                if layer == "l3":
                    intervention_type = payload.get("intervention_type", "none")
                    if str(intervention_type).lower() != "none":
                        l3_interventions += 1

        summary["missing_fields"][layer] = {k: v for k, v in layer_missing.items() if v > 0}

    l2_steps = summary["step_count"]["l2"]
    l3_steps = summary["step_count"]["l3"]
    summary["action_saturation_rate"] = float(l2_saturated) / float(l2_steps) if l2_steps else 0.0
    summary["intervention_rate"] = float(l3_interventions) / float(l3_steps) if l3_steps else 0.0

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V5.1 layer logs")
    parser.add_argument("logs_root", help="Path to logs root that contains l1/l2/l3 folders")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    summary = summarize_logs(Path(args.logs_root))
    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
