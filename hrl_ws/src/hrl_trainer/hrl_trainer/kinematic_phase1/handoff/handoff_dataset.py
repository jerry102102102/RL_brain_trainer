"""JSONL helpers and summary utilities for Phase 1C handoff datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POSITION_BUCKETS = (0.005, 0.008, 0.012, 0.020, 0.030, 0.050)
ORIENTATION_BUCKETS = (0.25, 0.50, 1.00, 1.50, 2.00, 3.20)
ACTION_BUCKETS = (0.10, 0.20, 0.30, 0.50, 0.80, 1.20)


def bucketize(value: float, edges: Iterable[float], *, unit: str = "") -> str:
    value = float(value)
    prev = 0.0
    for edge in edges:
        if value <= edge:
            return f"{prev:.3f}-{edge:.3f}{unit}"
        prev = float(edge)
    return f">{prev:.3f}{unit}"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record)) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _hist(values: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(v) for v in values))


def summarize_handoff_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos_buckets = [r.get("position_error_bucket", bucketize(r.get("position_error", 0.0), POSITION_BUCKETS, unit="m")) for r in records]
    ori_buckets = [
        r.get("orientation_error_bucket", bucketize(r.get("orientation_error", 0.0), ORIENTATION_BUCKETS, unit="rad"))
        for r in records
    ]
    dwell_counts = [int(r.get("dwell_count", 0)) for r in records]
    action_buckets = [
        r.get("action_magnitude_bucket", bucketize(r.get("action_magnitude", 0.0), ACTION_BUCKETS)) for r in records
    ]
    return {
        "total_samples": len(records),
        "position_bucket_counts": _hist(pos_buckets),
        "orientation_bucket_counts": _hist(ori_buckets),
        "dwell_count_distribution": _hist(dwell_counts),
        "action_magnitude_bucket_counts": _hist(action_buckets),
        "rule_ready_count": int(sum(1 for r in records if r.get("is_switch_rule_ready"))),
        "rule_ready_fraction": float(np.mean([bool(r.get("is_switch_rule_ready")) for r in records])) if records else 0.0,
        "mean_position_error": float(np.mean([float(r.get("position_error", 0.0)) for r in records])) if records else 0.0,
        "mean_orientation_error": float(np.mean([float(r.get("orientation_error", 0.0)) for r in records])) if records else 0.0,
        "mean_action_magnitude": float(np.mean([float(r.get("action_magnitude", 0.0)) for r in records])) if records else 0.0,
    }


def summarize_labeled_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize_handoff_records(records)
    labels = [bool(r.get("dock_success_from_here", False)) for r in records]
    base.update(
        {
            "dock_success_count": int(sum(labels)),
            "dock_success_rate": float(np.mean(labels)) if labels else 0.0,
            "dock_strict_pose_hit_rate": float(np.mean([bool(r.get("dock_strict_pose_hit_from_here")) for r in records]))
            if records
            else 0.0,
            "dock_dwell_success_rate": float(np.mean([bool(r.get("dock_dwell_success_from_here")) for r in records]))
            if records
            else 0.0,
            "mean_dock_final_position_error": float(
                np.mean([float(r.get("dock_final_position_error_from_here", 0.0)) for r in records])
            )
            if records
            else 0.0,
            "mean_dock_final_orientation_error": float(
                np.mean([float(r.get("dock_final_orientation_error_from_here", 0.0)) for r in records])
            )
            if records
            else 0.0,
        }
    )
    for key in ("position_error_bucket", "orientation_error_bucket", "dwell_count", "is_switch_rule_ready"):
        grouped: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(key))].append(bool(record.get("dock_success_from_here", False)))
        base[f"dock_success_rate_by_{key}"] = {
            name: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0}
            for name, vals in sorted(grouped.items())
        }
    return base


__all__ = [
    "ACTION_BUCKETS",
    "ORIENTATION_BUCKETS",
    "POSITION_BUCKETS",
    "bucketize",
    "read_jsonl",
    "summarize_handoff_records",
    "summarize_labeled_records",
    "to_jsonable",
    "write_jsonl",
]
