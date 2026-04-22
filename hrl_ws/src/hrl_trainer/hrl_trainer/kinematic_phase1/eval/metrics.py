"""Evaluation metrics for Phase 1 deterministic rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EvalConfig:
    suite_seed: int = 700001
    episodes: int = 10
    regression_tolerance_m: float = 0.01


@dataclass(frozen=True)
class EpisodeEvalMetrics:
    episode_id: int
    success: bool
    pre_near_goal_hit: bool
    near_goal_hit: bool
    dwell_success: bool
    regression: bool
    final_position_error: float
    final_orientation_error: float
    min_position_error: float
    final_minus_min_position_error: float
    action_l2_mean: float


def summarize_episode_metrics(metrics: Sequence[EpisodeEvalMetrics]) -> dict[str, float | int]:
    if not metrics:
        return {
            "episode_count": 0,
            "success_rate": 0.0,
            "pre_near_goal_hit_rate": 0.0,
            "near_goal_hit_rate": 0.0,
            "dwell_success_rate": 0.0,
            "regression_rate": 0.0,
            "mean_final_position_error": 0.0,
            "mean_final_orientation_error": 0.0,
            "mean_final_minus_min_position_error": 0.0,
            "average_action_magnitude": 0.0,
        }

    final_pos = np.asarray([m.final_position_error for m in metrics], dtype=float)
    final_ori = np.asarray([m.final_orientation_error for m in metrics], dtype=float)
    final_minus_min = np.asarray([m.final_minus_min_position_error for m in metrics], dtype=float)
    action_l2 = np.asarray([m.action_l2_mean for m in metrics], dtype=float)
    return {
        "episode_count": len(metrics),
        "success_rate": float(np.mean([m.success for m in metrics])),
        "pre_near_goal_hit_rate": float(np.mean([m.pre_near_goal_hit for m in metrics])),
        "near_goal_hit_rate": float(np.mean([m.near_goal_hit for m in metrics])),
        "dwell_success_rate": float(np.mean([m.dwell_success for m in metrics])),
        "regression_rate": float(np.mean([m.regression for m in metrics])),
        "mean_final_position_error": float(np.mean(final_pos)),
        "mean_final_orientation_error": float(np.mean(final_ori)),
        "mean_final_minus_min_position_error": float(np.mean(final_minus_min)),
        "average_action_magnitude": float(np.mean(action_l2)),
    }


def episode_metrics_to_jsonable(metrics: Sequence[EpisodeEvalMetrics]) -> list[dict[str, object]]:
    return [asdict(item) for item in metrics]
