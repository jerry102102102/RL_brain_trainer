"""Adaptive bucket prioritization for full workspace coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BucketPriority:
    bucket_id: str
    success_rate: float
    mean_min_error: float
    mean_final_error: float
    previous_success_rate: float | None
    failure_count: int
    category: str
    sampling_priority: float


def classify_bucket(
    *,
    success_rate: float,
    mean_min_error: float,
    mean_final_error: float,
    previous_success_rate: float | None = None,
) -> str:
    if previous_success_rate is not None and previous_success_rate >= 0.75 and success_rate < previous_success_rate - 0.20:
        return "forgetting_risk"
    if success_rate >= 0.85:
        return "mastered"
    if 0.35 <= success_rate < 0.85:
        return "frontier"
    if success_rate < 0.20 and mean_min_error > 0.025:
        return "too_hard"
    if mean_min_error <= 0.012 and mean_final_error > mean_min_error + 0.006:
        return "hard_but_promising"
    return "stress"


def priority_for_category(category: str) -> float:
    return {
        "mastered": 0.15,
        "frontier": 1.00,
        "hard_but_promising": 0.95,
        "forgetting_risk": 1.10,
        "stress": 0.25,
        "too_hard": 0.05,
    }.get(category, 0.20)


def update_bucket_priorities(bucket_metrics: dict[str, dict[str, Any]]) -> list[BucketPriority]:
    priorities: list[BucketPriority] = []
    for bucket_id, metrics in bucket_metrics.items():
        success_rate = float(metrics.get("success_rate", 0.0))
        mean_min_error = float(metrics.get("mean_min_position_error", metrics.get("mean_min_error", 999.0)))
        mean_final_error = float(metrics.get("mean_final_position_error", metrics.get("mean_final_error", 999.0)))
        previous = metrics.get("previous_success_rate")
        previous_success_rate = float(previous) if previous is not None else None
        category = classify_bucket(
            success_rate=success_rate,
            mean_min_error=mean_min_error,
            mean_final_error=mean_final_error,
            previous_success_rate=previous_success_rate,
        )
        base = priority_for_category(category)
        failure_count = int(metrics.get("failure_count", 0))
        priorities.append(
            BucketPriority(
                bucket_id=bucket_id,
                success_rate=success_rate,
                mean_min_error=mean_min_error,
                mean_final_error=mean_final_error,
                previous_success_rate=previous_success_rate,
                failure_count=failure_count,
                category=category,
                sampling_priority=float(base * (1.0 + min(failure_count, 20) / 40.0)),
            )
        )
    return sorted(priorities, key=lambda item: item.sampling_priority, reverse=True)
