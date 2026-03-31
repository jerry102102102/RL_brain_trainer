from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EPISODE_REQUIRED_KEYS = {
    "episode_index",
    "seed",
    "return_total",
    "terminal_reason",
    "success",
    "episode_length",
    "min_goal_error_per_episode",
    "final_goal_error",
    "near_goal_dwell_steps",
    "goal_hit_steps",
    "safety_violation_count",
    "intervention_count",
    "action_l2_mean",
    "action_l2_p95",
    "action_clamp_ratio_mean",
    "jerk_l2_mean",
    "timeout_flag",
    "checkpoint_path",
}

SUMMARY_REQUIRED_KEYS = {
    "success_rate",
    "median_return",
    "p50_episode_length",
    "p95_episode_length",
    "median_min_goal_error",
    "p90_near_goal_dwell_steps",
    "safety_abort_rate",
    "intervention_rate",
    "gate_decision",
    "gate_reasons",
}

TRAIN_REQUIRED_KEYS = {
    "global_step",
    "critic_loss_1",
    "critic_loss_2",
    "actor_loss",
    "alpha",
    "alpha_loss",
    "q_target_mean",
    "q_target_std",
    "replay_size",
}


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    missing_keys: tuple[str, ...]


def _check_keys(payload: dict[str, Any], required: set[str]) -> SchemaCheckResult:
    missing = tuple(sorted(required - set(payload.keys())))
    return SchemaCheckResult(ok=not missing, missing_keys=missing)


def validate_episode_row(row: dict[str, Any]) -> SchemaCheckResult:
    return _check_keys(row, EPISODE_REQUIRED_KEYS)


def validate_summary(summary: dict[str, Any]) -> SchemaCheckResult:
    return _check_keys(summary, SUMMARY_REQUIRED_KEYS)


def validate_train_row(row: dict[str, Any]) -> SchemaCheckResult:
    return _check_keys(row, TRAIN_REQUIRED_KEYS)
