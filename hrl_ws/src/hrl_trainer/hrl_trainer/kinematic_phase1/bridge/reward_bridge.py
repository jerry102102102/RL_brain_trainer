"""Bridge reward for cleaning dirty handoff states into Dock-acceptable states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class BridgeRewardConfig:
    position_keep_radius_m: float = 0.030
    position_progress_weight: float = 1.0
    orientation_progress_weight: float = 4.0
    orientation_reward_requires_position: bool = True
    position_keep_bonus: float = 0.0
    position_soft_keep_weight: float = 0.0
    orientation_center_weight: float = 0.0
    orientation_milestone_thresholds_rad: tuple[float, ...] = ()
    orientation_milestone_bonuses: tuple[float, ...] = ()
    realign_return_enabled: bool = False
    coarse_orientation_threshold_rad: float = 1.0
    coarse_orientation_bonus: float = 0.0
    return_position_progress_weight: float = 0.0
    return_position_center_weight: float = 0.0
    return_position_bonus: float = 0.0
    return_orientation_progress_weight: float = 0.0
    motion_cleanup_weight: float = 0.5
    action_magnitude_weight: float = 0.01
    action_delta_weight: float = 0.02
    leave_near_goal_penalty: float = 1.0
    terminate_on_leave_near_goal: bool = False
    position_regression_weight: float = 2.0
    orientation_regression_weight: float = 6.0
    joint_limit_penalty_weight: float = 0.05
    acceptance_region_bonus: float = 1.5
    acceptance_pos_threshold_m: float = 0.008
    acceptance_ori_threshold_rad: float = 1.0
    success_bonus: float = 2.0


def compute_bridge_reward(
    *,
    prev_pose6: Sequence[float],
    curr_pose6: Sequence[float],
    goal_pose6: Sequence[float],
    action: Sequence[float],
    prev_action: Sequence[float],
    dq_norm: float,
    joint_limit_margin_min: float = 1.0,
    config: BridgeRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or BridgeRewardConfig()
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)
    prev_pos = l2_norm(prev_pos_err)
    curr_pos = l2_norm(curr_pos_err)
    prev_ori = l2_norm(prev_ori_err)
    curr_ori = l2_norm(curr_ori_err)
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    in_acceptance = curr_pos <= cfg.acceptance_pos_threshold_m and curr_ori <= cfg.acceptance_ori_threshold_rad
    left_near_goal = curr_pos > cfg.position_keep_radius_m
    orientation_gate = (not cfg.orientation_reward_requires_position) or curr_pos <= cfg.position_keep_radius_m
    in_return_phase = cfg.realign_return_enabled and curr_ori <= cfg.coarse_orientation_threshold_rad
    coarse_orientation_hit = curr_ori <= cfg.coarse_orientation_threshold_rad
    return_position_hit = in_return_phase and curr_pos <= cfg.acceptance_pos_threshold_m
    position_progress_weight = cfg.return_position_progress_weight if in_return_phase else cfg.position_progress_weight
    orientation_progress_weight = (
        cfg.return_orientation_progress_weight if in_return_phase else cfg.orientation_progress_weight
    )
    milestone_bonus = 0.0
    for threshold, bonus in zip(cfg.orientation_milestone_thresholds_rad, cfg.orientation_milestone_bonuses, strict=False):
        if curr_ori <= float(threshold):
            milestone_bonus += float(bonus)
    components = {
        "position_progress": position_progress_weight * (prev_pos - curr_pos),
        "orientation_progress": orientation_progress_weight * (prev_ori - curr_ori) if orientation_gate else 0.0,
        "orientation_center": -cfg.orientation_center_weight * curr_ori if orientation_gate else 0.0,
        "orientation_milestone_bonus": milestone_bonus if orientation_gate else 0.0,
        "position_keep_bonus": cfg.position_keep_bonus if curr_pos <= cfg.position_keep_radius_m else 0.0,
        "position_soft_keep_penalty": -cfg.position_soft_keep_weight * max(curr_pos - cfg.position_keep_radius_m, 0.0),
        "coarse_orientation_bonus": cfg.coarse_orientation_bonus if coarse_orientation_hit else 0.0,
        "return_position_bonus": cfg.return_position_bonus if return_position_hit else 0.0,
        "return_position_center": -cfg.return_position_center_weight * curr_pos if in_return_phase else 0.0,
        "motion_cleanup": -cfg.motion_cleanup_weight * float(dq_norm),
        "smoothness_penalty": -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
        - cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2)),
        "leave_near_goal_penalty": -cfg.leave_near_goal_penalty if left_near_goal else 0.0,
        "position_regression_penalty": -cfg.position_regression_weight * max(curr_pos - prev_pos, 0.0),
        "orientation_regression_penalty": -cfg.orientation_regression_weight * max(curr_ori - prev_ori, 0.0),
        "joint_limit_penalty": -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25),
        "acceptance_region_bonus": cfg.acceptance_region_bonus if in_acceptance else 0.0,
        "success_bonus": cfg.success_bonus if in_acceptance else 0.0,
        "curr_pos_error": curr_pos,
        "curr_ori_error": curr_ori,
        "bridge_phase": 1.0 if in_return_phase else 0.0,
        "coarse_orientation_hit": float(coarse_orientation_hit),
        "return_position_hit": float(return_position_hit),
        "in_acceptance_region": float(in_acceptance),
    }
    reward = sum(
        components[key]
        for key in (
            "position_progress",
            "orientation_progress",
            "orientation_center",
            "orientation_milestone_bonus",
            "position_keep_bonus",
            "position_soft_keep_penalty",
            "coarse_orientation_bonus",
            "return_position_bonus",
            "return_position_center",
            "motion_cleanup",
            "smoothness_penalty",
            "leave_near_goal_penalty",
            "position_regression_penalty",
            "orientation_regression_penalty",
            "joint_limit_penalty",
            "acceptance_region_bonus",
            "success_bonus",
        )
    )
    return float(reward), components


__all__ = ["BridgeRewardConfig", "compute_bridge_reward"]
