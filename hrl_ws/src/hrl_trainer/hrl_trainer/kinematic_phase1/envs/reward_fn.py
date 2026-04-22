"""Simple, interpretable Phase 1 reward function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class RewardConfig:
    position_progress_weight: float = 8.0
    orientation_progress_weight: float = 0.0
    pre_near_goal_pos_threshold_m: float = 0.12
    near_goal_pos_threshold_m: float = 0.05
    near_goal_ori_threshold_rad: float = 0.35
    use_orientation_gate: bool = False
    pre_near_goal_bonus: float = 0.03
    near_goal_bonus: float = 0.10
    dwell_bonus: float = 0.12
    drift_penalty_weight: float = 3.0
    action_magnitude_weight: float = 0.002
    action_delta_weight: float = 0.004
    joint_limit_penalty_weight: float = 0.05
    success_bonus: float = 1.0


def compute_reward(
    *,
    prev_pose6: Sequence[float],
    curr_pose6: Sequence[float],
    goal_pose6: Sequence[float],
    action: Sequence[float],
    prev_action: Sequence[float],
    curr_in_pre_near_goal: bool,
    prev_in_near_goal: bool,
    curr_in_near_goal: bool,
    dwell_count: int,
    joint_limit_margin_min: float,
    success: bool,
    config: RewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or RewardConfig()
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)

    prev_pos_norm = l2_norm(prev_pos_err)
    curr_pos_norm = l2_norm(curr_pos_err)
    prev_ori_norm = l2_norm(prev_ori_err)
    curr_ori_norm = l2_norm(curr_ori_err)

    position_progress = cfg.position_progress_weight * (prev_pos_norm - curr_pos_norm)
    orientation_progress = cfg.orientation_progress_weight * (prev_ori_norm - curr_ori_norm)
    pre_near_goal = cfg.pre_near_goal_bonus if curr_in_pre_near_goal and not curr_in_near_goal else 0.0
    near_goal = cfg.near_goal_bonus if curr_in_near_goal and not prev_in_near_goal else 0.0
    dwell = cfg.dwell_bonus if curr_in_near_goal and dwell_count >= 2 else 0.0
    drift_penalty = -cfg.drift_penalty_weight * max(curr_pos_norm - prev_pos_norm, 0.0) if prev_in_near_goal else 0.0
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    smoothness_penalty = -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
    smoothness_penalty += -cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    joint_limit_penalty = -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25)
    success_bonus = cfg.success_bonus if success else 0.0

    components = {
        "position_progress": float(position_progress),
        "orientation_progress": float(orientation_progress),
        "pre_near_goal_bonus": float(pre_near_goal),
        "near_goal_bonus": float(near_goal),
        "dwell_bonus": float(dwell),
        "drift_penalty": float(drift_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "joint_limit_penalty": float(joint_limit_penalty),
        "success_bonus": float(success_bonus),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "dwell_count": float(dwell_count),
        "in_near_goal": float(curr_in_near_goal),
    }
    reward = sum(
        components[name]
        for name in (
            "position_progress",
            "orientation_progress",
            "pre_near_goal_bonus",
            "near_goal_bonus",
            "dwell_bonus",
            "drift_penalty",
            "smoothness_penalty",
            "joint_limit_penalty",
            "success_bonus",
        )
    )
    return float(reward), components
