"""Start-target pair utilities for random-start workspace coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairDifficultyThresholds:
    local_q_l2: float = 0.28
    medium_q_l2: float = 0.70
    frontier_success_low: float = 0.35
    frontier_success_high: float = 0.80


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def classify_pair(
    *,
    start: dict[str, Any],
    target: dict[str, Any],
    q_l2: float,
    thresholds: PairDifficultyThresholds = PairDifficultyThresholds(),
) -> str:
    target_success = target.get("previous_eval_success_rate")
    if start.get("source_type") in {"home", "successful_rollout"} and target.get("stage_id") is not None and int(target["stage_id"]) <= 7:
        return "retention"
    if q_l2 <= thresholds.local_q_l2:
        return "local"
    if target_success is not None:
        success = float(target_success)
        if thresholds.frontier_success_low <= success <= thresholds.frontier_success_high:
            return "frontier"
        if success < thresholds.frontier_success_low:
            return "stress"
    if q_l2 <= thresholds.medium_q_l2:
        return "medium"
    return "frontier" if int(target.get("stage_id") or 0) <= 10 else "stress"


def build_pair_sampler_summary(
    *,
    start_map_path: Path,
    target_map_path: Path,
    seed: int,
    pair_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    starts = load_jsonl(start_map_path)
    targets = load_jsonl(target_map_path)
    rng = np.random.default_rng(seed)
    pairs: list[dict[str, Any]] = []
    if not starts or not targets:
        return [], {"pair_count": 0, "reason": "empty start or target map"}

    for idx in range(max(pair_count, 0)):
        start = starts[int(rng.integers(0, len(starts)))]
        target = targets[int(rng.integers(0, len(targets)))]
        q_start = np.asarray(start["q_start"], dtype=float)
        q_target = np.asarray(target["q_target"], dtype=float)
        start_pos = np.asarray(start["ee_position"], dtype=float)
        target_pos = np.asarray(target["ee_target_position"], dtype=float)
        start_ori = np.asarray(start["ee_orientation"], dtype=float)
        target_ori = np.asarray(target["ee_target_orientation"], dtype=float)
        q_l2 = float(np.linalg.norm(q_target - q_start))
        ee_pos_l2 = float(np.linalg.norm(target_pos - start_pos))
        ori_l2 = float(np.linalg.norm(target_ori - start_ori))
        difficulty_class = classify_pair(start=start, target=target, q_l2=q_l2)
        pairs.append(
            {
                "pair_id": f"pair_{idx:06d}",
                "start_id": start["start_id"],
                "target_id": target["target_id"],
                "start_source_type": start.get("source_type"),
                "target_source_type": target.get("source_type"),
                "target_stage_id": target.get("stage_id"),
                "start_bucket_id": start.get("bucket_id"),
                "target_bucket_id": target.get("bucket_id"),
                "joint_distance_l2": q_l2,
                "ee_position_distance": ee_pos_l2,
                "orientation_distance": ori_l2,
                "z_displacement": float(abs(target_pos[2] - start_pos[2])),
                "start_joint_limit_margin": float(start.get("joint_limit_margin_min", 0.0)),
                "target_joint_limit_margin": float(target.get("joint_limit_margin_min", 0.0)),
                "difficulty_class": difficulty_class,
            }
        )

    class_counts: dict[str, int] = {}
    for pair in pairs:
        key = str(pair["difficulty_class"])
        class_counts[key] = class_counts.get(key, 0) + 1
    summary = {
        "seed": int(seed),
        "pair_count": len(pairs),
        "start_count": len(starts),
        "target_count": len(targets),
        "difficulty_class_counts": class_counts,
        "mean_joint_distance_l2": float(np.mean([p["joint_distance_l2"] for p in pairs])) if pairs else 0.0,
        "mean_ee_position_distance": float(np.mean([p["ee_position_distance"] for p in pairs])) if pairs else 0.0,
        "max_joint_distance_l2": float(max((p["joint_distance_l2"] for p in pairs), default=0.0)),
        "pair_curriculum_note": "Pairs are classified for layered curriculum; full-random stress pairs should remain a minority during training.",
    }
    return pairs, summary


def write_pair_sampler_outputs(pairs: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "start_target_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + ("\n" if pairs else ""))
    (output_dir / "pair_sampler_summary.json").write_text(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build random-start start-target pair summary.")
    parser.add_argument("--start-map", required=True)
    parser.add_argument("--target-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=930001)
    parser.add_argument("--pair-count", type=int, default=2048)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pairs, summary = build_pair_sampler_summary(
        start_map_path=Path(args.start_map),
        target_map_path=Path(args.target_map),
        seed=args.seed,
        pair_count=args.pair_count,
    )
    write_pair_sampler_outputs(pairs, summary, Path(args.output_dir))
    print(json.dumps({"output_dir": str(Path(args.output_dir)), **summary}, indent=2))


if __name__ == "__main__":
    main()
