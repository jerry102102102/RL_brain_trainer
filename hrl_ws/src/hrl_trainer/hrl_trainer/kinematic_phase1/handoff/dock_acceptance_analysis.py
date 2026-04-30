"""Analysis helpers for Dock acceptance maps."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .handoff_dataset import read_jsonl, write_jsonl


def bucket_label(lo: float, hi: float, unit: str) -> str:
    return f"{lo:.3f}-{hi:.3f}{unit}"


def summarize_acceptance_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_pos: dict[str, list[bool]] = defaultdict(list)
    grouped_ori: dict[str, list[bool]] = defaultdict(list)
    grouped_dq: dict[str, list[bool]] = defaultdict(list)
    grouped_prev_action: dict[str, list[bool]] = defaultdict(list)
    matrix: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    pair_success: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        success = bool(record.get("dock_success_from_here", False))
        pos_bucket = str(record.get("position_bucket"))
        ori_bucket = str(record.get("orientation_bucket"))
        grouped_pos[pos_bucket].append(success)
        grouped_ori[ori_bucket].append(success)
        grouped_dq[str(record.get("dq_bucket", "0.000"))].append(success)
        grouped_prev_action[str(record.get("prev_action_bucket", "0.000"))].append(success)
        pair_success[(pos_bucket, ori_bucket)].append(success)
    for (pos_bucket, ori_bucket), values in pair_success.items():
        matrix[pos_bucket][ori_bucket] = {"count": len(values), "success_rate": float(np.mean(values)) if values else 0.0}

    successes = [bool(r.get("dock_success_from_here", False)) for r in records]
    success_records = [r for r in records if r.get("dock_success_from_here", False)]
    return {
        "total_samples": len(records),
        "dock_success_count": int(sum(successes)),
        "dock_success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_rate_by_position_bucket": {
            key: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0}
            for key, vals in sorted(grouped_pos.items())
        },
        "success_rate_by_orientation_bucket": {
            key: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0}
            for key, vals in sorted(grouped_ori.items())
        },
        "success_matrix_position_by_orientation": {key: matrix[key] for key in sorted(matrix)},
        "success_rate_by_dq_bucket": {
            key: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0}
            for key, vals in sorted(grouped_dq.items())
        },
        "success_rate_by_prev_action_bucket": {
            key: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0}
            for key, vals in sorted(grouped_prev_action.items())
        },
        "mean_success_position_error": float(np.mean([float(r["perturbed_position_error"]) for r in success_records]))
        if success_records
        else None,
        "mean_success_orientation_error": float(np.mean([float(r["perturbed_orientation_error"]) for r in success_records]))
        if success_records
        else None,
        "max_success_position_error": float(max([float(r["perturbed_position_error"]) for r in success_records]))
        if success_records
        else None,
        "max_success_orientation_error": float(max([float(r["perturbed_orientation_error"]) for r in success_records]))
        if success_records
        else None,
    }


def write_acceptance_heatmap(records: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pos_labels = sorted({str(r.get("position_bucket")) for r in records})
    ori_labels = sorted({str(r.get("orientation_bucket")) for r in records})
    if not pos_labels or not ori_labels:
        return None
    values = np.full((len(pos_labels), len(ori_labels)), np.nan)
    for i, pos in enumerate(pos_labels):
        for j, ori in enumerate(ori_labels):
            subset = [bool(r.get("dock_success_from_here", False)) for r in records if r.get("position_bucket") == pos and r.get("orientation_bucket") == ori]
            if subset:
                values[i, j] = float(np.mean(subset))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(ori_labels) * 1.2), max(4, len(pos_labels) * 0.8)))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(ori_labels)), labels=ori_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pos_labels)), labels=pos_labels)
    ax.set_xlabel("orientation error bucket")
    ax.set_ylabel("position error bucket")
    ax.set_title("Dock acceptance success rate")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def load_acceptance_map(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def save_acceptance_map(path: Path, records: list[dict[str, Any]]) -> int:
    return write_jsonl(path, records)


__all__ = [
    "bucket_label",
    "load_acceptance_map",
    "save_acceptance_map",
    "summarize_acceptance_records",
    "write_acceptance_heatmap",
]
