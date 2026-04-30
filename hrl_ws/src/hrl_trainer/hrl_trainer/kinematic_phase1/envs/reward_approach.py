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
    orientation_milestone_thresholds_rad: tuple[float, ...] = ()
    orientation_milestone_bonuses: tuple[float, ...] = ()
    near_field_orientation_center_weight: float = 0.0
    use_orientation_gate: bool = False
    pre_near_goal_bonus: float = 0.03
    near_goal_bonus: float = 0.10
    near_goal_bonus_decay: float = 0.5
    pre_near_to_near_progress_weight: float = 0.0
    coarse_orientation_bonus: float = 0.04
    handover_pos_threshold_m: float = 0.0
    handover_ori_threshold_rad: float = 0.0
    handover_bonus: float = 0.0
    handover_retention_bonus: float = 0.0
    handover_dwell_bonus: float = 0.0
    handover_leave_penalty: float = 0.0
    handover_regression_weight: float = 0.0
    handover_smoothness_multiplier: float = 1.0
    dock_coarse_ready_pos_threshold_m: float = 0.0
    dock_coarse_ready_ori_threshold_rad: float = 0.0
    dock_coarse_ready_action_threshold: float = 0.0
    dock_coarse_ready_dq_threshold: float = 0.0
    dock_coarse_ready_bonus: float = 0.0
    dock_coarse_ready_retention_bonus: float = 0.0
    dock_coarse_ready_dwell_bonus: float = 0.0
    dock_coarse_ready_leave_penalty: float = 0.0
    dock_coarse_ready_regression_weight: float = 0.0
    finisher_ready_pos_threshold_m: float = 0.0
    finisher_ready_ori_threshold_rad: float = 0.0
    finisher_ready_action_threshold: float = 0.0
    finisher_ready_dq_threshold: float = 0.0
    finisher_ready_bonus: float = 0.0
    finisher_ready_retention_bonus: float = 0.0
    finisher_ready_dwell_bonus: float = 0.0
    finisher_ready_leave_penalty: float = 0.0
    finisher_ready_regression_weight: float = 0.0
    near_handoff_pos_threshold_m: float = 0.0
    near_handoff_ori_threshold_rad: float = 0.0
    near_handoff_action_weight: float = 0.0
    near_handoff_dq_weight: float = 0.0
    near_handoff_motion_bonus_weight: float = 0.0
    near_handoff_settle_bonus_weight: float = 0.0
    same_step_alignment_bonus: float = 0.0
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
    dq_norm: float = 0.0,
    prev_dq_norm: float = 0.0,
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
    orientation_milestone_bonus = 0.0
    if curr_in_pre_near_goal:
        for threshold, bonus in zip(cfg.orientation_milestone_thresholds_rad, cfg.orientation_milestone_bonuses, strict=False):
            if curr_ori_norm <= float(threshold):
                orientation_milestone_bonus += float(bonus)
    near_field_orientation_center = -cfg.near_field_orientation_center_weight * curr_ori_norm if curr_in_pre_near_goal else 0.0

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
    handover_retention_bonus = cfg.handover_retention_bonus if curr_in_handover_zone and prev_in_handover_zone else 0.0
    handover_dwell_bonus = cfg.handover_dwell_bonus if curr_in_handover_zone and dwell_count >= 2 else 0.0
    handover_leave_penalty = -cfg.handover_leave_penalty if prev_in_handover_zone and not curr_in_handover_zone else 0.0
    handover_regression_penalty = (
        -cfg.handover_regression_weight * (max(curr_pos_norm - prev_pos_norm, 0.0) + max(curr_ori_norm - prev_ori_norm, 0.0))
        if prev_in_handover_zone or curr_in_handover_zone
        else 0.0
    )
    dwell = cfg.dwell_bonus if curr_in_near_goal and dwell_count >= 2 else 0.0
    drift_escalation_count = max(int(near_goal_drift_count) - int(cfg.drift_penalty_escalation_start), 0)
    drift_penalty_scale = 1.0 + cfg.drift_penalty_escalation_per_count * drift_escalation_count
    drift_penalty_weight = cfg.drift_penalty_weight * drift_penalty_scale
    drift_penalty = -drift_penalty_weight * max(curr_pos_norm - prev_pos_norm, 0.0) if prev_in_near_goal else 0.0
    near_goal_leave_penalty = -cfg.near_goal_leave_penalty if prev_in_near_goal and not curr_in_near_goal else 0.0
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    action_norm = float(np.linalg.norm(action_arr))
    prev_action_norm = float(np.linalg.norm(prev_action_arr))
    dq_norm_f = float(dq_norm)
    prev_dq_norm_f = float(prev_dq_norm)
    dc_ready_enabled = (
        cfg.dock_coarse_ready_pos_threshold_m > 0.0
        and cfg.dock_coarse_ready_ori_threshold_rad > 0.0
    )
    curr_dc_ready_pose = (
        dc_ready_enabled
        and curr_pos_norm <= cfg.dock_coarse_ready_pos_threshold_m
        and curr_ori_norm <= cfg.dock_coarse_ready_ori_threshold_rad
    )
    prev_dc_ready_pose = (
        dc_ready_enabled
        and prev_pos_norm <= cfg.dock_coarse_ready_pos_threshold_m
        and prev_ori_norm <= cfg.dock_coarse_ready_ori_threshold_rad
    )
    curr_dc_motion_ready = (
        (cfg.dock_coarse_ready_action_threshold <= 0.0 or action_norm <= cfg.dock_coarse_ready_action_threshold)
        and (cfg.dock_coarse_ready_dq_threshold <= 0.0 or dq_norm_f <= cfg.dock_coarse_ready_dq_threshold)
    )
    prev_dc_motion_ready = (
        (cfg.dock_coarse_ready_action_threshold <= 0.0 or prev_action_norm <= cfg.dock_coarse_ready_action_threshold)
        and (cfg.dock_coarse_ready_dq_threshold <= 0.0 or prev_dq_norm_f <= cfg.dock_coarse_ready_dq_threshold)
    )
    curr_in_dc_ready = bool(curr_dc_ready_pose and curr_dc_motion_ready)
    prev_in_dc_ready = bool(prev_dc_ready_pose and prev_dc_motion_ready)
    finisher_ready_enabled = (
        cfg.finisher_ready_pos_threshold_m > 0.0
        and cfg.finisher_ready_ori_threshold_rad > 0.0
    )
    curr_finisher_ready_pose = (
        finisher_ready_enabled
        and curr_pos_norm <= cfg.finisher_ready_pos_threshold_m
        and curr_ori_norm <= cfg.finisher_ready_ori_threshold_rad
    )
    prev_finisher_ready_pose = (
        finisher_ready_enabled
        and prev_pos_norm <= cfg.finisher_ready_pos_threshold_m
        and prev_ori_norm <= cfg.finisher_ready_ori_threshold_rad
    )
    curr_finisher_motion_ready = (
        (cfg.finisher_ready_action_threshold <= 0.0 or action_norm <= cfg.finisher_ready_action_threshold)
        and (cfg.finisher_ready_dq_threshold <= 0.0 or dq_norm_f <= cfg.finisher_ready_dq_threshold)
    )
    prev_finisher_motion_ready = (
        (cfg.finisher_ready_action_threshold <= 0.0 or prev_action_norm <= cfg.finisher_ready_action_threshold)
        and (cfg.finisher_ready_dq_threshold <= 0.0 or prev_dq_norm_f <= cfg.finisher_ready_dq_threshold)
    )
    curr_in_finisher_ready = bool(curr_finisher_ready_pose and curr_finisher_motion_ready)
    prev_in_finisher_ready = bool(prev_finisher_ready_pose and prev_finisher_motion_ready)
    near_handoff_zone = (
        cfg.near_handoff_pos_threshold_m > 0.0
        and cfg.near_handoff_ori_threshold_rad > 0.0
        and curr_pos_norm <= cfg.near_handoff_pos_threshold_m
        and curr_ori_norm <= cfg.near_handoff_ori_threshold_rad
    )
    prev_near_handoff_zone = (
        cfg.near_handoff_pos_threshold_m > 0.0
        and cfg.near_handoff_ori_threshold_rad > 0.0
        and prev_pos_norm <= cfg.near_handoff_pos_threshold_m
        and prev_ori_norm <= cfg.near_handoff_ori_threshold_rad
    )
    dc_ready_bonus = cfg.dock_coarse_ready_bonus if curr_in_dc_ready and not prev_in_dc_ready else 0.0
    dc_ready_retention_bonus = cfg.dock_coarse_ready_retention_bonus if curr_in_dc_ready and prev_in_dc_ready else 0.0
    dc_ready_dwell_bonus = cfg.dock_coarse_ready_dwell_bonus if curr_in_dc_ready and dwell_count >= 2 else 0.0
    dc_ready_leave_penalty = -cfg.dock_coarse_ready_leave_penalty if prev_in_dc_ready and not curr_in_dc_ready else 0.0
    dc_ready_regression_penalty = (
        -cfg.dock_coarse_ready_regression_weight
        * (max(curr_pos_norm - prev_pos_norm, 0.0) + max(curr_ori_norm - prev_ori_norm, 0.0))
        if near_handoff_zone or prev_near_handoff_zone or curr_dc_ready_pose or prev_dc_ready_pose
        else 0.0
    )
    finisher_ready_bonus = cfg.finisher_ready_bonus if curr_in_finisher_ready and not prev_in_finisher_ready else 0.0
    finisher_ready_retention_bonus = cfg.finisher_ready_retention_bonus if curr_in_finisher_ready and prev_in_finisher_ready else 0.0
    finisher_ready_dwell_bonus = cfg.finisher_ready_dwell_bonus if curr_in_finisher_ready and dwell_count >= 2 else 0.0
    finisher_ready_leave_penalty = -cfg.finisher_ready_leave_penalty if prev_in_finisher_ready and not curr_in_finisher_ready else 0.0
    finisher_ready_regression_penalty = (
        -cfg.finisher_ready_regression_weight
        * (max(curr_pos_norm - prev_pos_norm, 0.0) + max(curr_ori_norm - prev_ori_norm, 0.0))
        if near_handoff_zone
        or prev_near_handoff_zone
        or curr_finisher_ready_pose
        or prev_finisher_ready_pose
        else 0.0
    )
    near_handoff_action_penalty = (
        -cfg.near_handoff_action_weight * float(np.mean(action_arr**2))
        if near_handoff_zone or curr_dc_ready_pose or curr_finisher_ready_pose
        else 0.0
    )
    near_handoff_dq_penalty = (
        -cfg.near_handoff_dq_weight * dq_norm_f if near_handoff_zone or curr_dc_ready_pose or curr_finisher_ready_pose else 0.0
    )
    near_handoff_motion_bonus = 0.0
    if near_handoff_zone or curr_dc_ready_pose or curr_finisher_ready_pose:
        action_threshold = cfg.finisher_ready_action_threshold or cfg.dock_coarse_ready_action_threshold
        dq_threshold = cfg.finisher_ready_dq_threshold or cfg.dock_coarse_ready_dq_threshold
        action_scale = max(action_threshold, 1e-9)
        dq_scale = max(dq_threshold, 1e-9)
        action_clean = max(1.0 - action_norm / action_scale, 0.0) if action_threshold > 0 else 0.0
        dq_clean = max(1.0 - dq_norm_f / dq_scale, 0.0) if dq_threshold > 0 else 0.0
        near_handoff_motion_bonus = cfg.near_handoff_motion_bonus_weight * (0.5 * action_clean + 0.5 * dq_clean)
    near_handoff_settle_bonus = 0.0
    if near_handoff_zone or curr_dc_ready_pose or curr_finisher_ready_pose:
        near_handoff_settle_bonus = cfg.near_handoff_settle_bonus_weight * (
            0.5 * max(prev_action_norm - action_norm, 0.0) + 0.5 * max(prev_dq_norm_f - dq_norm_f, 0.0)
        )
    same_step_alignment_bonus = (
        cfg.same_step_alignment_bonus
        if curr_pos_norm < prev_pos_norm and curr_ori_norm < prev_ori_norm and (curr_in_pre_near_goal or near_handoff_zone)
        else 0.0
    )
    smoothness_multiplier = cfg.handover_smoothness_multiplier if curr_in_handover_zone or prev_in_handover_zone else 1.0
    smoothness_penalty = smoothness_multiplier * (
        -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
        - cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    )
    joint_limit_penalty = -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25)
    success_bonus = cfg.success_bonus if success else 0.0

    components = {
        "position_progress": float(position_progress),
        "global_orientation_progress": float(global_orientation_progress),
        "near_field_orientation_progress": float(near_field_orientation_progress),
        "orientation_progress": float(orientation_progress),
        "orientation_milestone_bonus": float(orientation_milestone_bonus),
        "near_field_orientation_center": float(near_field_orientation_center),
        "pre_near_goal_bonus": float(pre_near_goal),
        "near_goal_bonus": float(near_goal),
        "pre_near_to_near_progress": float(inner_progress),
        "near_goal_bonus_scale": float(near_goal_bonus_scale if curr_in_near_goal and not prev_in_near_goal else 0.0),
        "coarse_orientation_bonus": float(coarse_orientation_bonus),
        "handover_bonus": float(handover_bonus),
        "handover_retention_bonus": float(handover_retention_bonus),
        "handover_dwell_bonus": float(handover_dwell_bonus),
        "handover_leave_penalty": float(handover_leave_penalty),
        "handover_regression_penalty": float(handover_regression_penalty),
        "dock_coarse_ready_bonus": float(dc_ready_bonus),
        "dock_coarse_ready_retention_bonus": float(dc_ready_retention_bonus),
        "dock_coarse_ready_dwell_bonus": float(dc_ready_dwell_bonus),
        "dock_coarse_ready_leave_penalty": float(dc_ready_leave_penalty),
        "dock_coarse_ready_regression_penalty": float(dc_ready_regression_penalty),
        "finisher_ready_bonus": float(finisher_ready_bonus),
        "finisher_ready_retention_bonus": float(finisher_ready_retention_bonus),
        "finisher_ready_dwell_bonus": float(finisher_ready_dwell_bonus),
        "finisher_ready_leave_penalty": float(finisher_ready_leave_penalty),
        "finisher_ready_regression_penalty": float(finisher_ready_regression_penalty),
        "near_handoff_action_penalty": float(near_handoff_action_penalty),
        "near_handoff_dq_penalty": float(near_handoff_dq_penalty),
        "near_handoff_motion_bonus": float(near_handoff_motion_bonus),
        "near_handoff_settle_bonus": float(near_handoff_settle_bonus),
        "same_step_alignment_bonus": float(same_step_alignment_bonus),
        "dwell_bonus": float(dwell),
        "drift_penalty": float(drift_penalty),
        "near_goal_leave_penalty": float(near_goal_leave_penalty),
        "drift_penalty_scale": float(drift_penalty_scale),
        "near_goal_entry_count": float(near_goal_entry_count),
        "near_goal_drift_count": float(near_goal_drift_count),
        "smoothness_penalty": float(smoothness_penalty),
        "smoothness_multiplier": float(smoothness_multiplier),
        "joint_limit_penalty": float(joint_limit_penalty),
        "success_bonus": float(success_bonus),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "curr_action_norm": float(action_norm),
        "curr_dq_norm": float(dq_norm_f),
        "dwell_count": float(dwell_count),
        "in_pre_near_goal": float(curr_in_pre_near_goal),
        "in_near_goal": float(curr_in_near_goal),
        "in_handover_zone": float(curr_in_handover_zone),
        "in_dock_coarse_ready": float(curr_in_dc_ready),
        "in_dock_coarse_ready_pose": float(curr_dc_ready_pose),
        "in_finisher_ready": float(curr_in_finisher_ready),
        "in_finisher_ready_pose": float(curr_finisher_ready_pose),
        "in_near_handoff_zone": float(near_handoff_zone),
    }
    reward = sum(
        components[name]
        for name in (
            "position_progress",
            "orientation_progress",
            "orientation_milestone_bonus",
            "near_field_orientation_center",
            "pre_near_goal_bonus",
            "near_goal_bonus",
            "pre_near_to_near_progress",
            "coarse_orientation_bonus",
            "handover_bonus",
            "handover_retention_bonus",
            "handover_dwell_bonus",
            "handover_leave_penalty",
            "handover_regression_penalty",
            "dock_coarse_ready_bonus",
            "dock_coarse_ready_retention_bonus",
            "dock_coarse_ready_dwell_bonus",
            "dock_coarse_ready_leave_penalty",
            "dock_coarse_ready_regression_penalty",
            "finisher_ready_bonus",
            "finisher_ready_retention_bonus",
            "finisher_ready_dwell_bonus",
            "finisher_ready_leave_penalty",
            "finisher_ready_regression_penalty",
            "near_handoff_action_penalty",
            "near_handoff_dq_penalty",
            "near_handoff_motion_bonus",
            "near_handoff_settle_bonus",
            "same_step_alignment_bonus",
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
