"""Termination and truncation checks for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass(frozen=True)
class TerminationConfig:
    max_episode_steps: int = 75
    success_pos_threshold_m: float = 0.06
    success_ori_threshold_rad: float = 0.15
    success_dwell_steps: int = 2
    require_orientation: bool = False
    terminate_on_success: bool = True


def evaluate_termination(
    *,
    step_count: int,
    pos_error_norm: float,
    ori_error_norm: float,
    dwell_count: int,
    invalid_state: bool = False,
    config: TerminationConfig | None = None,
) -> dict[str, bool | str]:
    cfg = config or TerminationConfig()
    terminated = False
    truncated = False
    success = False
    reason = "running"
    success_criteria_met = (
        pos_error_norm <= cfg.success_pos_threshold_m
        and (not cfg.require_orientation or ori_error_norm <= cfg.success_ori_threshold_rad)
        and dwell_count >= cfg.success_dwell_steps
    )

    if invalid_state or not math.isfinite(pos_error_norm) or not math.isfinite(ori_error_norm):
        terminated = True
        reason = "invalid_state"
    elif success_criteria_met:
        success = True
        if cfg.terminate_on_success:
            terminated = True
            reason = "success"
    if not terminated and step_count >= cfg.max_episode_steps:
        truncated = True
        reason = "max_steps"

    return {
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
        "reason": reason,
    }
