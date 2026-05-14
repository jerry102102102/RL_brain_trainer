"""Sequential gate evaluator for route curriculum candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..training.policy_config import write_json
from .eval_route_curriculum import evaluate_sequential_route


def _passes_prefix120(summary: dict[str, Any], *, min_success_rate: float) -> bool:
    return int(summary.get("longest_success_prefix", 0)) >= 120 and float(summary.get("success_rate", 0.0)) >= min_success_rate


def evaluate_route_gate(
    *,
    checkpoint_path: Path,
    config_path: Path,
    route_path: Path,
    artifact_root: Path,
    prefixes: list[int],
    full_end_index: int | None,
    min_prefix120_success_rate: float,
    best_full_longest_prefix: int,
    full_prefix_tolerance: int,
) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    prefix_results: dict[str, Any] = {}
    for prefix in prefixes:
        prefix_results[f"prefix_{prefix}"] = evaluate_sequential_route(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            route_path=route_path,
            artifact_root=artifact_root / f"prefix_{prefix}",
            start_index=1,
            end_index=int(prefix),
        )

    full_summary = None
    if full_end_index is not None:
        full_summary = evaluate_sequential_route(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            route_path=route_path,
            artifact_root=artifact_root / f"full_{full_end_index}",
            start_index=1,
            end_index=int(full_end_index),
        )

    p120 = prefix_results.get("prefix_120")
    p180 = prefix_results.get("prefix_180")
    prefix120_retained = bool(p120 and _passes_prefix120(p120, min_success_rate=min_prefix120_success_rate))
    expands_beyond_120 = bool(p180 and int(p180.get("longest_success_prefix", 0)) > 120)
    first_failure_not_before_120 = bool(
        p180
        and (
            p180.get("first_failure_index") is None
            or int(p180.get("first_failure_index", 0)) > 120
        )
    )
    full_not_too_regressed = True
    if full_summary is not None:
        full_not_too_regressed = int(full_summary.get("longest_success_prefix", 0)) >= int(best_full_longest_prefix - full_prefix_tolerance)

    accepted = bool(prefix120_retained and expands_beyond_120 and first_failure_not_before_120 and full_not_too_regressed)
    reasons: list[str] = []
    if not prefix120_retained:
        reasons.append("prefix120_retention_failed")
    if not expands_beyond_120:
        reasons.append("prefix180_did_not_expand_beyond_120")
    if not first_failure_not_before_120:
        reasons.append("prefix180_failed_before_or_at_120")
    if not full_not_too_regressed:
        reasons.append("full_route_prefix_regressed_too_much")

    summary = {
        "schema_version": "v5.route_gate.v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "route_path": str(route_path),
        "accepted": accepted,
        "rejection_reasons": reasons,
        "criteria": {
            "min_prefix120_success_rate": float(min_prefix120_success_rate),
            "best_full_longest_prefix": int(best_full_longest_prefix),
            "full_prefix_tolerance": int(full_prefix_tolerance),
        },
        "prefix_results": prefix_results,
        "full_result": full_summary,
    }
    write_json(artifact_root / "route_gate_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential gate checks for route candidate checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--prefixes", default="20,40,80,120,180")
    parser.add_argument("--full-end-index", type=int)
    parser.add_argument("--min-prefix120-success-rate", type=float, default=0.98)
    parser.add_argument("--best-full-longest-prefix", type=int, default=170)
    parser.add_argument("--full-prefix-tolerance", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prefixes = [int(part.strip()) for part in str(args.prefixes).split(",") if part.strip()]
    evaluate_route_gate(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        route_path=Path(args.route_path),
        artifact_root=Path(args.artifact_root),
        prefixes=prefixes,
        full_end_index=args.full_end_index,
        min_prefix120_success_rate=args.min_prefix120_success_rate,
        best_full_longest_prefix=args.best_full_longest_prefix,
        full_prefix_tolerance=args.full_prefix_tolerance,
    )


if __name__ == "__main__":
    main()
