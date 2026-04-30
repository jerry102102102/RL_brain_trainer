"""Compare real approach handoff states against the Dock acceptance basin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..training.policy_config import write_json
from .handoff_dataset import read_jsonl


def _bucket_success_table(acceptance_records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float | int]]:
    table: dict[tuple[str, str], list[bool]] = {}
    for record in acceptance_records:
        key = (str(record.get("position_bucket")), str(record.get("orientation_bucket")))
        table.setdefault(key, []).append(bool(record.get("dock_success_from_here", False)))
    return {key: {"count": len(vals), "success_rate": float(np.mean(vals)) if vals else 0.0} for key, vals in table.items()}


def _success_region_description(table: dict[tuple[str, str], dict[str, float | int]], threshold: float) -> list[dict[str, object]]:
    rows = []
    for (pos_bucket, ori_bucket), stats in sorted(table.items()):
        if float(stats["success_rate"]) >= threshold:
            rows.append({"position_bucket": pos_bucket, "orientation_bucket": ori_bucket, **stats})
    return rows


def compare_handoff_to_acceptance(
    *,
    handoff_dataset: Path,
    acceptance_map: Path,
    artifact_root: Path,
    success_threshold: float = 0.5,
) -> dict[str, object]:
    handoff_records = read_jsonl(handoff_dataset)
    acceptance_records = read_jsonl(acceptance_map)
    table = _bucket_success_table(acceptance_records)
    high_success_keys = {key for key, stats in table.items() if float(stats["success_rate"]) >= success_threshold}
    overlap_flags = [
        (str(record.get("position_error_bucket")), str(record.get("orientation_error_bucket"))) in high_success_keys
        for record in handoff_records
    ]
    accepted_records = [record for record in acceptance_records if record.get("dock_success_from_here", False)]
    handoff_mean_pos = float(np.mean([float(r.get("position_error", 0.0)) for r in handoff_records])) if handoff_records else 0.0
    handoff_mean_ori = float(np.mean([float(r.get("orientation_error", 0.0)) for r in handoff_records])) if handoff_records else 0.0
    acceptance_mean_pos = (
        float(np.mean([float(r.get("perturbed_position_error", 0.0)) for r in accepted_records])) if accepted_records else None
    )
    acceptance_mean_ori = (
        float(np.mean([float(r.get("perturbed_orientation_error", 0.0)) for r in accepted_records])) if accepted_records else None
    )
    pos_gap = None if acceptance_mean_pos is None else handoff_mean_pos - acceptance_mean_pos
    ori_gap = None if acceptance_mean_ori is None else handoff_mean_ori - acceptance_mean_ori
    if pos_gap is None or ori_gap is None:
        primary_gap = "unknown_no_success_region"
    elif abs(ori_gap) > abs(pos_gap) * 20.0:
        primary_gap = "orientation"
    elif abs(pos_gap) > abs(ori_gap) / 20.0:
        primary_gap = "position"
    else:
        primary_gap = "mixed"
    summary = {
        "handoff_dataset": str(handoff_dataset),
        "acceptance_map": str(acceptance_map),
        "success_region_threshold": success_threshold,
        "handoff_count": len(handoff_records),
        "acceptance_count": len(acceptance_records),
        "acceptance_success_count": int(sum(1 for r in acceptance_records if r.get("dock_success_from_here", False))),
        "overlap_count": int(sum(overlap_flags)),
        "overlap_fraction": float(np.mean(overlap_flags)) if overlap_flags else 0.0,
        "handoff_mean_position_error": handoff_mean_pos,
        "handoff_mean_orientation_error": handoff_mean_ori,
        "acceptance_success_mean_position_error": acceptance_mean_pos,
        "acceptance_success_mean_orientation_error": acceptance_mean_ori,
        "estimated_position_gap": pos_gap,
        "estimated_orientation_gap": ori_gap,
        "estimated_primary_gap_dimension": primary_gap,
        "acceptance_success_region_description": _success_region_description(table, success_threshold),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "handoff_vs_acceptance_summary.json", summary)
    _write_overlay_plot(handoff_records, acceptance_records, artifact_root / "plots" / "handoff_vs_acceptance_scatter.png")
    return summary


def _write_overlay_plot(handoff_records: list[dict[str, Any]], acceptance_records: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    acc_x = [float(r.get("perturbed_position_error", 0.0)) for r in acceptance_records]
    acc_y = [float(r.get("perturbed_orientation_error", 0.0)) for r in acceptance_records]
    acc_c = [1.0 if r.get("dock_success_from_here", False) else 0.0 for r in acceptance_records]
    ax.scatter(acc_x, acc_y, c=acc_c, cmap="viridis", alpha=0.45, s=16, label="acceptance samples")
    handoff_x = [float(r.get("position_error", 0.0)) for r in handoff_records]
    handoff_y = [float(r.get("orientation_error", 0.0)) for r in handoff_records]
    ax.scatter(handoff_x, handoff_y, c="tab:red", alpha=0.35, s=10, label="approach handoff")
    ax.set_xlabel("position error (m)")
    ax.set_ylabel("orientation error (rad)")
    ax.set_title("Approach handoff states over Dock acceptance samples")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Phase 1C handoff states against Dock acceptance map.")
    parser.add_argument("--handoff-dataset", required=True)
    parser.add_argument("--acceptance-map", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    summary = compare_handoff_to_acceptance(
        handoff_dataset=Path(args.handoff_dataset),
        acceptance_map=Path(args.acceptance_map),
        artifact_root=Path(args.artifact_root),
        success_threshold=args.success_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
