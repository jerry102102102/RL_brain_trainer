"""Dock-policy reward with stronger stabilize-and-hold shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class DockRewardConfig:
    position_progress_weight: float = 6.0
    orientation_progress_weight: float = 5.0
    stay_in_zone_bonus: float = 0.08
    dwell_bonus: float = 0.18
    leave_zone_penalty: float = 0.25
    drift_penalty_position_weight: float = 4.0
    drift_penalty_orientation_weight: float = 2.0
    action_magnitude_weight: float = 0.006
    action_delta_weight: float = 0.012
    joint_limit_penalty_weight: float = 0.05
    success_bonus: float = 2.0
    tight_pose_pos_threshold_m: float = 0.005
    tight_pose_ori_threshold_rad: float = 0.05
    tight_pose_bonus: float = 0.0
    tight_pose_dwell_bonus: float = 0.0
    strict_pose_leave_penalty: float = 0.0
    strict_center_reward_weight: float = 0.0
    strict_center_position_weight: float = 0.0
    strict_center_orientation_weight: float = 0.0
    strict_center_small_action_bonus_weight: float = 0.0
    strict_center_small_action_pos_radius_m: float = 0.0
    strict_center_small_action_ori_radius_rad: float = 0.0
    strict_center_small_action_scale: float = 0.0
    strict_center_small_action_power: float = 2.0
    strict_zone_drift_penalty_multiplier: float = 1.0
    strict_zone_action_penalty_multiplier: float = 1.0
    tight_position_shaping_radius_m: float = 0.0
    tight_position_shaping_weight: float = 0.0
    tight_orientation_shaping_radius_rad: float = 0.0
    tight_orientation_shaping_weight: float = 0.0
    convergence_position_radius_m: float = 0.0
    convergence_position_progress_weight: float = 0.0
    convergence_orientation_radius_rad: float = 0.0
    convergence_orientation_progress_weight: float = 0.0
    position_first_orientation_pos_threshold_m: float = 0.0
    position_first_orientation_pre_scale: float = 1.0


def compute_dock_reward(
    *,
    prev_pose6: Sequence[float],
    curr_pose6: Sequence[float],
    goal_pose6: Sequence[float],
    action: Sequence[float],
    prev_action: Sequence[float],
    prev_in_near_goal: bool,
    curr_in_near_goal: bool,
    dwell_count: int,
    joint_limit_margin_min: float,
    success: bool,
    near_goal_entry_count: int = 0,
    near_goal_drift_count: int = 0,
    config: DockRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or DockRewardConfig()
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)

    prev_pos_norm = l2_norm(prev_pos_err)
    curr_pos_norm = l2_norm(curr_pos_err)
    prev_ori_norm = l2_norm(prev_ori_err)
    curr_ori_norm = l2_norm(curr_ori_err)

    position_progress = cfg.position_progress_weight * (prev_pos_norm - curr_pos_norm)
    orientation_progress = cfg.orientation_progress_weight * (prev_ori_norm - curr_ori_norm)
    stay_in_zone = cfg.stay_in_zone_bonus if curr_in_near_goal else 0.0
    dwell_bonus = cfg.dwell_bonus * max(dwell_count - 1, 0) if curr_in_near_goal else 0.0
    curr_in_tight_pose = curr_pos_norm <= cfg.tight_pose_pos_threshold_m and curr_ori_norm <= cfg.tight_pose_ori_threshold_rad
    prev_in_tight_pose = prev_pos_norm <= cfg.tight_pose_pos_threshold_m and prev_ori_norm <= cfg.tight_pose_ori_threshold_rad
    strict_pos_closeness = max(1.0 - curr_pos_norm / max(cfg.tight_pose_pos_threshold_m, 1e-9), 0.0)
    strict_ori_closeness = max(1.0 - curr_ori_norm / max(cfg.tight_pose_ori_threshold_rad, 1e-9), 0.0)
    strict_closeness = (0.8 * strict_pos_closeness + 0.2 * strict_ori_closeness) ** 2
    tight_pose_bonus = cfg.tight_pose_bonus if curr_in_tight_pose else 0.0
    tight_pose_dwell_bonus = cfg.tight_pose_dwell_bonus * max(dwell_count - 1, 0) if curr_in_tight_pose else 0.0
    strict_pose_leave_penalty = -cfg.strict_pose_leave_penalty if prev_in_tight_pose and not curr_in_tight_pose else 0.0
    strict_center_reward = cfg.strict_center_reward_weight * strict_closeness if curr_in_tight_pose else 0.0
    strict_center_position_penalty = (
        -cfg.strict_center_position_weight * (curr_pos_norm / max(cfg.tight_pose_pos_threshold_m, 1e-9)) ** 2
        if cfg.strict_center_position_weight > 0.0
        else 0.0
    )
    strict_center_orientation_penalty = (
        -cfg.strict_center_orientation_weight * (curr_ori_norm / max(cfg.tight_pose_ori_threshold_rad, 1e-9)) ** 2
        if cfg.strict_center_orientation_weight > 0.0
        else 0.0
    )
    action_rms = float(np.sqrt(np.mean(np.asarray(action, dtype=float) ** 2)))
    strict_center_small_action_bonus = 0.0
    if (
        cfg.strict_center_small_action_bonus_weight > 0.0
        and cfg.strict_center_small_action_pos_radius_m > 0.0
        and cfg.strict_center_small_action_ori_radius_rad > 0.0
        and cfg.strict_center_small_action_scale > 0.0
    ):
        center_pos_closeness = max(1.0 - curr_pos_norm / cfg.strict_center_small_action_pos_radius_m, 0.0)
        center_ori_closeness = max(1.0 - curr_ori_norm / cfg.strict_center_small_action_ori_radius_rad, 0.0)
        center_closeness = (0.8 * center_pos_closeness + 0.2 * center_ori_closeness) ** cfg.strict_center_small_action_power
        action_smallness = max(1.0 - action_rms / cfg.strict_center_small_action_scale, 0.0)
        strict_center_small_action_bonus = (
            cfg.strict_center_small_action_bonus_weight * center_closeness * action_smallness
            if curr_in_tight_pose
            else 0.0
        )
    tight_position_shaping = (
        cfg.tight_position_shaping_weight * max(1.0 - curr_pos_norm / max(cfg.tight_position_shaping_radius_m, 1e-9), 0.0)
        if cfg.tight_position_shaping_radius_m > 0.0
        else 0.0
    )
    tight_orientation_shaping = (
        cfg.tight_orientation_shaping_weight
        * max(1.0 - curr_ori_norm / max(cfg.tight_orientation_shaping_radius_rad, 1e-9), 0.0)
        if cfg.tight_orientation_shaping_radius_rad > 0.0
        else 0.0
    )
    convergence_position_progress = (
        cfg.convergence_position_progress_weight * (prev_pos_norm - curr_pos_norm)
        if cfg.convergence_position_radius_m > 0.0
        and min(prev_pos_norm, curr_pos_norm) <= cfg.convergence_position_radius_m
        else 0.0
    )
    orientation_position_gate_scale = (
        cfg.position_first_orientation_pre_scale
        if cfg.position_first_orientation_pos_threshold_m > 0.0
        and curr_pos_norm > cfg.position_first_orientation_pos_threshold_m
        else 1.0
    )
    convergence_orientation_progress = (
        orientation_position_gate_scale * cfg.convergence_orientation_progress_weight * (prev_ori_norm - curr_ori_norm)
        if cfg.convergence_orientation_radius_rad > 0.0
        and min(prev_ori_norm, curr_ori_norm) <= cfg.convergence_orientation_radius_rad
        else 0.0
    )
    leave_zone_penalty = -cfg.leave_zone_penalty if prev_in_near_goal and not curr_in_near_goal else 0.0
    drift_penalty = -cfg.drift_penalty_position_weight * max(curr_pos_norm - prev_pos_norm, 0.0)
    drift_penalty += -cfg.drift_penalty_orientation_weight * max(curr_ori_norm - prev_ori_norm, 0.0)
    if curr_in_tight_pose or prev_in_tight_pose:
        drift_penalty *= cfg.strict_zone_drift_penalty_multiplier

    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    smoothness_penalty = -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
    smoothness_penalty += -cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    if curr_in_tight_pose:
        smoothness_penalty *= cfg.strict_zone_action_penalty_multiplier
    joint_limit_penalty = -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25)
    success_bonus = cfg.success_bonus if success else 0.0

    components = {
        "position_progress": float(position_progress),
        "orientation_progress": float(orientation_progress),
        "stay_in_zone_bonus": float(stay_in_zone),
        "dwell_bonus": float(dwell_bonus),
        "tight_pose_bonus": float(tight_pose_bonus),
        "tight_pose_dwell_bonus": float(tight_pose_dwell_bonus),
        "strict_pose_leave_penalty": float(strict_pose_leave_penalty),
        "strict_center_reward": float(strict_center_reward),
        "strict_center_position_penalty": float(strict_center_position_penalty),
        "strict_center_orientation_penalty": float(strict_center_orientation_penalty),
        "strict_center_small_action_bonus": float(strict_center_small_action_bonus),
        "tight_position_shaping": float(tight_position_shaping),
        "tight_orientation_shaping": float(tight_orientation_shaping),
        "convergence_position_progress": float(convergence_position_progress),
        "convergence_orientation_progress": float(convergence_orientation_progress),
        "orientation_position_gate_scale": float(orientation_position_gate_scale),
        "leave_zone_penalty": float(leave_zone_penalty),
        "drift_penalty": float(drift_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "joint_limit_penalty": float(joint_limit_penalty),
        "success_bonus": float(success_bonus),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "dwell_count": float(dwell_count),
        "in_tight_pose": float(curr_in_tight_pose),
        "near_goal_entry_count": float(near_goal_entry_count),
        "near_goal_drift_count": float(near_goal_drift_count),
        "in_near_goal": float(curr_in_near_goal),
    }
    reward = sum(
        components[name]
        for name in (
            "position_progress",
            "orientation_progress",
            "stay_in_zone_bonus",
            "dwell_bonus",
            "tight_pose_bonus",
            "tight_pose_dwell_bonus",
            "strict_pose_leave_penalty",
            "strict_center_reward",
            "strict_center_position_penalty",
            "strict_center_orientation_penalty",
            "strict_center_small_action_bonus",
            "tight_position_shaping",
            "tight_orientation_shaping",
            "convergence_position_progress",
            "convergence_orientation_progress",
            "leave_zone_penalty",
            "drift_penalty",
            "smoothness_penalty",
            "joint_limit_penalty",
            "success_bonus",
        )
    )
    return float(reward), components
