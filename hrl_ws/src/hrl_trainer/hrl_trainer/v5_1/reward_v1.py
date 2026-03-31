from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TerminalReason = Literal["success", "timeout", "safety_abort", "intervention_abort", "ongoing"]


@dataclass(frozen=True)
class RewardV1Config:
    near_goal_tol: float = 0.050
    goal_tol_pos: float = 0.020
    goal_tol_yaw: float = 0.08

    w_progress: float = 1.2
    w_near: float = 0.8
    w_goal_hit: float = 2.0
    w_action: float = 0.03
    w_jerk: float = 0.02
    w_safety: float = 1.5
    w_clamp: float = 0.6
    w_intervention: float = 2.5

    terminal_success_bonus: float = 8.0
    terminal_timeout_penalty: float = 3.0
    terminal_safety_penalty: float = 6.0


@dataclass(frozen=True)
class RewardV1Breakdown:
    reward_step: float
    reward_terminal: float
    reward_total: float
    progress: float
    near: float
    goal_hit: float
    action_delta_l2: float
    jerk_l2: float
    safety_violation: float
    clamp_ratio: float
    intervention: float
    terminal_reason: TerminalReason


def _l2(vec: np.ndarray | None) -> float:
    if vec is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float32), ord=2))


def compute_reward_v1(
    *,
    error_prev: float | None,
    error_curr: float,
    yaw_error_curr: float,
    action_curr: np.ndarray,
    action_prev: np.ndarray | None,
    action_prev2: np.ndarray | None,
    clamp_ratio: float,
    safety_violation_event: bool,
    intervention_event: bool,
    terminal_reason: TerminalReason = "ongoing",
    config: RewardV1Config | None = None,
) -> RewardV1Breakdown:
    cfg = config or RewardV1Config()

    progress = 0.0
    if error_prev is not None:
        denom = max(float(error_prev), 1e-6)
        progress = float(np.clip((float(error_prev) - float(error_curr)) / denom, -1.0, 1.0))

    near = 1.0 if float(error_curr) <= cfg.near_goal_tol else 0.0
    goal_hit = 1.0 if (float(error_curr) <= cfg.goal_tol_pos and float(abs(yaw_error_curr)) <= cfg.goal_tol_yaw) else 0.0

    action_delta = 0.0
    if action_prev is not None:
        action_delta = _l2(np.asarray(action_curr) - np.asarray(action_prev))

    jerk = 0.0
    if action_prev is not None and action_prev2 is not None:
        jerk = _l2(np.asarray(action_curr) - 2.0 * np.asarray(action_prev) + np.asarray(action_prev2))

    safety = 1.0 if safety_violation_event else 0.0
    intervention = 1.0 if intervention_event else 0.0
    clamp = float(np.clip(clamp_ratio, 0.0, 1.0))

    step_reward = (
        cfg.w_progress * progress
        + cfg.w_near * near
        + cfg.w_goal_hit * goal_hit
        - cfg.w_action * action_delta
        - cfg.w_jerk * jerk
        - cfg.w_safety * safety
        - cfg.w_clamp * clamp
        - cfg.w_intervention * intervention
    )

    terminal_reward = 0.0
    if terminal_reason == "success":
        terminal_reward += cfg.terminal_success_bonus
    elif terminal_reason == "timeout":
        terminal_reward -= cfg.terminal_timeout_penalty
    elif terminal_reason in {"safety_abort", "intervention_abort"}:
        terminal_reward -= cfg.terminal_safety_penalty

    total = step_reward + terminal_reward
    return RewardV1Breakdown(
        reward_step=float(step_reward),
        reward_terminal=float(terminal_reward),
        reward_total=float(total),
        progress=float(progress),
        near=float(near),
        goal_hit=float(goal_hit),
        action_delta_l2=float(action_delta),
        jerk_l2=float(jerk),
        safety_violation=float(safety),
        clamp_ratio=float(clamp),
        intervention=float(intervention),
        terminal_reason=terminal_reason,
    )
