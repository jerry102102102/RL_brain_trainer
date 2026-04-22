"""Approach-policy reward for Phase 1B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class ApproachRewardConfig:
    position_progress_weight: float = 8.0
    orientation_progress_weight: float = 1.0
    near_field_orientation_progress_weight: float = 2.0
    pre_near_goal_pos_threshold_m: float = 0.12
    near_goal_pos_threshold_m: float = 0.05
    near_goal_ori_threshold_rad: float = 0.35
    coarse_orientation_bonus_threshold_rad: float = 0.35
    use_orientation_gate: bool = False
    pre_near_goal_bonus: float = 0.03
    near_goal_bonus: float = 0.10
    near_goal_bonus_decay: float = 0.5
    pre_near_to_near_progress_weight: float = 0.0
    coarse_orientation_bonus: float = 0.04
    handover_pos_threshold_m: float = 0.0
    handover_ori_threshold_rad: float = 0.0
    handover_bonus: float = 0.0
    handover_dwell_bonus: float = 0.0
    dwell_bonus: float = 0.12
    drift_penalty_weight: float = 3.0
    drift_penalty_escalation_start: int = 2
    drift_penalty_escalation_per_count: float = 0.5
    near_goal_leave_penalty: float = 0.0
    action_magnitude_weight: float = 0.002
    action_delta_weight: float = 0.004
    joint_limit_penalty_weight: float = 0.05
    success_bonus: float = 1.0


def compute_approach_reward(
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
    near_goal_entry_count: int = 0,
    near_goal_drift_count: int = 0,
    config: ApproachRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or ApproachRewardConfig()
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)

    prev_pos_norm = l2_norm(prev_pos_err)
    curr_pos_norm = l2_norm(curr_pos_err)
    prev_ori_norm = l2_norm(prev_ori_err)
    curr_ori_norm = l2_norm(curr_ori_err)

    position_progress = cfg.position_progress_weight * (prev_pos_norm - curr_pos_norm)
    global_orientation_progress = cfg.orientation_progress_weight * (prev_ori_norm - curr_ori_norm)
    near_field_orientation_progress = (
        cfg.near_field_orientation_progress_weight * (prev_ori_norm - curr_ori_norm) if curr_in_pre_near_goal else 0.0
    )
    orientation_progress = global_orientation_progress + near_field_orientation_progress

    pre_near_goal = cfg.pre_near_goal_bonus if curr_in_pre_near_goal and not curr_in_near_goal else 0.0
    near_goal_bonus_scale = cfg.near_goal_bonus_decay ** max(int(near_goal_entry_count) - 1, 0)
    near_goal = cfg.near_goal_bonus * near_goal_bonus_scale if curr_in_near_goal and not prev_in_near_goal else 0.0
    inner_progress = (
        cfg.pre_near_to_near_progress_weight * max(prev_pos_norm - curr_pos_norm, 0.0)
        if curr_in_pre_near_goal and not curr_in_near_goal
        else 0.0
    )
    coarse_orientation_bonus = (
        cfg.coarse_orientation_bonus
        if curr_in_pre_near_goal and curr_ori_norm <= cfg.coarse_orientation_bonus_threshold_rad
        else 0.0
    )
    curr_in_handover_zone = (
        cfg.handover_pos_threshold_m > 0.0
        and curr_pos_norm <= cfg.handover_pos_threshold_m
        and (cfg.handover_ori_threshold_rad <= 0.0 or curr_ori_norm <= cfg.handover_ori_threshold_rad)
    )
    prev_in_handover_zone = (
        cfg.handover_pos_threshold_m > 0.0
        and prev_pos_norm <= cfg.handover_pos_threshold_m
        and (cfg.handover_ori_threshold_rad <= 0.0 or prev_ori_norm <= cfg.handover_ori_threshold_rad)
    )
    handover_bonus = cfg.handover_bonus if curr_in_handover_zone and not prev_in_handover_zone else 0.0
    handover_dwell_bonus = cfg.handover_dwell_bonus if curr_in_handover_zone and dwell_count >= 2 else 0.0
    dwell = cfg.dwell_bonus if curr_in_near_goal and dwell_count >= 2 else 0.0
    drift_escalation_count = max(int(near_goal_drift_count) - int(cfg.drift_penalty_escalation_start), 0)
    drift_penalty_scale = 1.0 + cfg.drift_penalty_escalation_per_count * drift_escalation_count
    drift_penalty_weight = cfg.drift_penalty_weight * drift_penalty_scale
    drift_penalty = -drift_penalty_weight * max(curr_pos_norm - prev_pos_norm, 0.0) if prev_in_near_goal else 0.0
    near_goal_leave_penalty = -cfg.near_goal_leave_penalty if prev_in_near_goal and not curr_in_near_goal else 0.0
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    smoothness_penalty = -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
    smoothness_penalty += -cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    joint_limit_penalty = -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25)
    success_bonus = cfg.success_bonus if success else 0.0

    components = {
        "position_progress": float(position_progress),
        "global_orientation_progress": float(global_orientation_progress),
        "near_field_orientation_progress": float(near_field_orientation_progress),
        "orientation_progress": float(orientation_progress),
        "pre_near_goal_bonus": float(pre_near_goal),
        "near_goal_bonus": float(near_goal),
        "pre_near_to_near_progress": float(inner_progress),
        "near_goal_bonus_scale": float(near_goal_bonus_scale if curr_in_near_goal and not prev_in_near_goal else 0.0),
        "coarse_orientation_bonus": float(coarse_orientation_bonus),
        "handover_bonus": float(handover_bonus),
        "handover_dwell_bonus": float(handover_dwell_bonus),
        "dwell_bonus": float(dwell),
        "drift_penalty": float(drift_penalty),
        "near_goal_leave_penalty": float(near_goal_leave_penalty),
        "drift_penalty_scale": float(drift_penalty_scale),
        "near_goal_entry_count": float(near_goal_entry_count),
        "near_goal_drift_count": float(near_goal_drift_count),
        "smoothness_penalty": float(smoothness_penalty),
        "joint_limit_penalty": float(joint_limit_penalty),
        "success_bonus": float(success_bonus),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "dwell_count": float(dwell_count),
        "in_pre_near_goal": float(curr_in_pre_near_goal),
        "in_near_goal": float(curr_in_near_goal),
        "in_handover_zone": float(curr_in_handover_zone),
    }
    reward = sum(
        components[name]
        for name in (
            "position_progress",
            "orientation_progress",
            "pre_near_goal_bonus",
            "near_goal_bonus",
            "pre_near_to_near_progress",
            "coarse_orientation_bonus",
            "handover_bonus",
            "handover_dwell_bonus",
            "dwell_bonus",
            "drift_penalty",
            "near_goal_leave_penalty",
            "smoothness_penalty",
            "joint_limit_penalty",
            "success_bonus",
        )
    )
    return float(reward), components


__all__ = ["ApproachRewardConfig", "compute_approach_reward"]
