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
    working_range_bonus: float = 0.0
    working_range_dwell_bonus: float = 0.0
    working_range_dwell_start: int = 2
    working_range_exit_penalty: float = 0.0
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
    strict_center_dwell_bonus_weight: float = 0.0
    strict_center_dwell_start: int = 2
    strict_center_dwell_escalation_start: int = 5
    strict_center_dwell_escalation_per_step: float = 0.0
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
    action_delta_violation_threshold: float = 0.0
    action_delta_violation_weight: float = 0.0
    delta_q_change_penalty_threshold: float = 0.0
    delta_q_change_penalty_weight: float = 0.0
    entry_action_penalty_near_pos_threshold_m: float = 0.0
    entry_action_penalty_far_pos_threshold_m: float = 0.0
    entry_action_penalty_near_multiplier: float = 1.0
    entry_action_penalty_far_multiplier: float = 1.0
    # V5.1-style basin shaping for handoff-to-dock end-to-end finetuning.
    # Disabled by default so previous dock experiments keep the same reward.
    basin_outer_radius_m: float = 0.0
    basin_inner_radius_m: float = 0.0
    basin_dwell_radius_m: float = 0.0
    basin_outer_bonus: float = 0.0
    basin_inner_bonus: float = 0.0
    basin_dwell_bonus: float = 0.0
    basin_outer_exit_penalty: float = 0.0
    basin_inner_exit_penalty: float = 0.0
    basin_dwell_break_penalty: float = 0.0
    basin_drift_penalty_weight: float = 0.0
    # Hold-and-preserve adaptation terms. Disabled by default so earlier dock
    # experiments remain comparable unless an adaptation config opts in.
    near_strict_pos_threshold_m: float = 0.0
    near_strict_ori_threshold_rad: float = 0.0
    preserve_state_bonus: float = 0.0
    preserve_position_tolerance_m: float = 0.0
    preserve_orientation_tolerance_rad: float = 0.0
    strict_hold_bonus: float = 0.0
    low_motion_bonus: float = 0.0
    low_motion_action_threshold: float = 0.0
    low_motion_dq_threshold: float = 0.0
    tiny_correction_bonus: float = 0.0
    tiny_correction_action_threshold: float = 0.0
    worse_than_entry_position_weight: float = 0.0
    worse_than_entry_orientation_weight: float = 0.0
    worse_than_entry_position_tolerance_m: float = 0.0
    worse_than_entry_orientation_tolerance_rad: float = 0.0
    near_strict_regression_multiplier: float = 1.0
    aggressive_action_weight: float = 0.0
    aggressive_action_threshold: float = 0.0
    dq_penalty_weight: float = 0.0
    dq_penalty_threshold: float = 0.0
    near_strict_action_penalty_multiplier: float = 1.0
    near_strict_dq_penalty_multiplier: float = 1.0


def _interpolate_entry_penalty_scale(
    *,
    pos_error_norm: float,
    near_threshold: float,
    far_threshold: float,
    near_multiplier: float,
    far_multiplier: float,
) -> float:
    if near_threshold <= 0.0 or far_threshold <= near_threshold:
        return 1.0
    if pos_error_norm <= near_threshold:
        return float(near_multiplier)
    if pos_error_norm >= far_threshold:
        return float(far_multiplier)
    alpha = (pos_error_norm - near_threshold) / max(far_threshold - near_threshold, 1e-9)
    return float(near_multiplier + alpha * (far_multiplier - near_multiplier))


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
    delta_q_change_l2: float = 0.0,
    dq_norm: float = 0.0,
    entry_pos_error_norm: float | None = None,
    entry_ori_error_norm: float | None = None,
    entry_action_l2: float | None = None,
    entry_dq_norm: float | None = None,
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
    working_range_bonus = cfg.working_range_bonus if curr_in_near_goal else 0.0
    working_range_dwell_bonus = (
        cfg.working_range_dwell_bonus * max(dwell_count - cfg.working_range_dwell_start + 1, 0)
        if curr_in_near_goal and dwell_count >= cfg.working_range_dwell_start
        else 0.0
    )
    curr_in_tight_pose = curr_pos_norm <= cfg.tight_pose_pos_threshold_m and curr_ori_norm <= cfg.tight_pose_ori_threshold_rad
    prev_in_tight_pose = prev_pos_norm <= cfg.tight_pose_pos_threshold_m and prev_ori_norm <= cfg.tight_pose_ori_threshold_rad
    near_strict_pos_threshold = cfg.near_strict_pos_threshold_m or cfg.tight_pose_pos_threshold_m * 2.0
    near_strict_ori_threshold = cfg.near_strict_ori_threshold_rad or cfg.tight_pose_ori_threshold_rad * 3.0
    curr_in_near_strict = curr_pos_norm <= near_strict_pos_threshold and curr_ori_norm <= near_strict_ori_threshold
    prev_in_near_strict = prev_pos_norm <= near_strict_pos_threshold and prev_ori_norm <= near_strict_ori_threshold
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
    strict_center_dwell_bonus = 0.0
    if curr_in_tight_pose and cfg.strict_center_dwell_bonus_weight > 0.0 and dwell_count >= cfg.strict_center_dwell_start:
        dwell_escalation_steps = max(dwell_count - cfg.strict_center_dwell_escalation_start, 0)
        dwell_scale = 1.0 + cfg.strict_center_dwell_escalation_per_step * dwell_escalation_steps
        strict_center_dwell_bonus = cfg.strict_center_dwell_bonus_weight * strict_closeness * dwell_scale
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
    working_range_exit_penalty = -cfg.working_range_exit_penalty if prev_in_near_goal and not curr_in_near_goal else 0.0
    drift_penalty = -cfg.drift_penalty_position_weight * max(curr_pos_norm - prev_pos_norm, 0.0)
    drift_penalty += -cfg.drift_penalty_orientation_weight * max(curr_ori_norm - prev_ori_norm, 0.0)
    if curr_in_tight_pose or prev_in_tight_pose:
        drift_penalty *= cfg.strict_zone_drift_penalty_multiplier

    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    action_l2 = float(np.linalg.norm(action_arr))
    entry_action_penalty_scale = _interpolate_entry_penalty_scale(
        pos_error_norm=max(prev_pos_norm, curr_pos_norm),
        near_threshold=cfg.entry_action_penalty_near_pos_threshold_m,
        far_threshold=cfg.entry_action_penalty_far_pos_threshold_m,
        near_multiplier=cfg.entry_action_penalty_near_multiplier,
        far_multiplier=cfg.entry_action_penalty_far_multiplier,
    )
    smoothness_penalty = -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
    smoothness_penalty += -cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    if curr_in_tight_pose:
        smoothness_penalty *= cfg.strict_zone_action_penalty_multiplier
    smoothness_penalty *= entry_action_penalty_scale
    action_delta_rms = float(np.sqrt(np.mean((action_arr - prev_action_arr) ** 2)))
    action_delta_violation_penalty = (
        -cfg.action_delta_violation_weight
        * entry_action_penalty_scale
        * max(action_delta_rms - cfg.action_delta_violation_threshold, 0.0)
        if cfg.action_delta_violation_weight > 0.0 and cfg.action_delta_violation_threshold > 0.0
        else 0.0
    )
    delta_q_change_penalty = (
        -cfg.delta_q_change_penalty_weight
        * entry_action_penalty_scale
        * max(float(delta_q_change_l2) - cfg.delta_q_change_penalty_threshold, 0.0)
        if cfg.delta_q_change_penalty_weight > 0.0 and cfg.delta_q_change_penalty_threshold > 0.0
        else 0.0
    )
    entry_pos = curr_pos_norm if entry_pos_error_norm is None else float(entry_pos_error_norm)
    entry_ori = curr_ori_norm if entry_ori_error_norm is None else float(entry_ori_error_norm)
    entry_action = action_l2 if entry_action_l2 is None else float(entry_action_l2)
    entry_dq = float(dq_norm) if entry_dq_norm is None else float(entry_dq_norm)
    preserve_state_bonus = 0.0
    if cfg.preserve_state_bonus > 0.0 and (curr_in_near_strict or curr_in_tight_pose):
        pos_not_worse = curr_pos_norm <= entry_pos + cfg.preserve_position_tolerance_m
        ori_not_worse = curr_ori_norm <= entry_ori + cfg.preserve_orientation_tolerance_rad
        if pos_not_worse and ori_not_worse:
            preserve_state_bonus = cfg.preserve_state_bonus
    strict_hold_bonus = cfg.strict_hold_bonus * max(dwell_count - 1, 0) if curr_in_tight_pose else 0.0
    low_motion_bonus = 0.0
    if (
        cfg.low_motion_bonus > 0.0
        and curr_in_near_strict
        and (cfg.low_motion_action_threshold <= 0.0 or action_l2 <= cfg.low_motion_action_threshold)
        and (cfg.low_motion_dq_threshold <= 0.0 or float(dq_norm) <= cfg.low_motion_dq_threshold)
    ):
        low_motion_bonus = cfg.low_motion_bonus
    tiny_correction_bonus = 0.0
    if cfg.tiny_correction_bonus > 0.0 and curr_in_near_strict and not curr_in_tight_pose:
        improved_pose = curr_pos_norm <= prev_pos_norm and curr_ori_norm <= prev_ori_norm
        small_action = cfg.tiny_correction_action_threshold <= 0.0 or action_l2 <= cfg.tiny_correction_action_threshold
        if improved_pose and small_action:
            tiny_correction_bonus = cfg.tiny_correction_bonus
    worse_than_entry_penalty = 0.0
    worse_than_entry_penalty += -cfg.worse_than_entry_position_weight * max(
        curr_pos_norm - entry_pos - cfg.worse_than_entry_position_tolerance_m,
        0.0,
    )
    worse_than_entry_penalty += -cfg.worse_than_entry_orientation_weight * max(
        curr_ori_norm - entry_ori - cfg.worse_than_entry_orientation_tolerance_rad,
        0.0,
    )
    near_strict_regression_penalty = 0.0
    if curr_in_near_strict or prev_in_near_strict:
        near_strict_regression_penalty = -cfg.near_strict_regression_multiplier * (
            cfg.drift_penalty_position_weight * max(curr_pos_norm - prev_pos_norm, 0.0)
            + cfg.drift_penalty_orientation_weight * max(curr_ori_norm - prev_ori_norm, 0.0)
        )
    aggressive_action_penalty_scale = cfg.near_strict_action_penalty_multiplier if curr_in_near_strict else 1.0
    aggressive_action_penalty = (
        -cfg.aggressive_action_weight
        * aggressive_action_penalty_scale
        * max(action_l2 - cfg.aggressive_action_threshold, 0.0)
        if cfg.aggressive_action_weight > 0.0 and cfg.aggressive_action_threshold > 0.0
        else 0.0
    )
    dq_penalty_scale = cfg.near_strict_dq_penalty_multiplier if curr_in_near_strict else 1.0
    dq_penalty = (
        -cfg.dq_penalty_weight * dq_penalty_scale * max(float(dq_norm) - cfg.dq_penalty_threshold, 0.0)
        if cfg.dq_penalty_weight > 0.0 and cfg.dq_penalty_threshold > 0.0
        else 0.0
    )
    entry_to_curr_delta_position_error = curr_pos_norm - entry_pos
    entry_to_curr_delta_orientation_error = curr_ori_norm - entry_ori
    entry_to_curr_delta_action_l2 = action_l2 - entry_action
    entry_to_curr_delta_dq_norm = float(dq_norm) - entry_dq
    joint_limit_penalty = -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25)
    success_bonus = cfg.success_bonus if success else 0.0

    basin_outer_bonus = 0.0
    basin_inner_bonus = 0.0
    basin_dwell_region_bonus = 0.0
    basin_outer_exit_penalty = 0.0
    basin_inner_exit_penalty = 0.0
    basin_dwell_break_penalty = 0.0
    basin_drift_penalty = 0.0
    basin_zone_index = 0
    if cfg.basin_outer_radius_m > 0.0 and cfg.basin_inner_radius_m > 0.0 and cfg.basin_dwell_radius_m > 0.0:
        outer_r = max(float(cfg.basin_outer_radius_m), 1e-9)
        inner_r = max(float(cfg.basin_inner_radius_m), 1e-9)
        dwell_r = max(float(cfg.basin_dwell_radius_m), 1e-9)

        prev_in_outer = prev_pos_norm <= outer_r
        prev_in_inner = prev_pos_norm <= inner_r
        prev_in_dwell_region = prev_pos_norm <= dwell_r
        curr_in_outer = curr_pos_norm <= outer_r
        curr_in_inner = curr_pos_norm <= inner_r
        curr_in_dwell_region = curr_pos_norm <= dwell_r
        basin_zone_index = 3 if curr_in_dwell_region else (2 if curr_in_inner else (1 if curr_in_outer else 0))

        if curr_in_outer:
            outer_closeness = max(1.0 - curr_pos_norm / outer_r, 0.0)
            basin_outer_bonus = cfg.basin_outer_bonus * (1.0 + outer_closeness)
        if curr_in_inner:
            inner_closeness = max(1.0 - curr_pos_norm / inner_r, 0.0)
            basin_inner_bonus = cfg.basin_inner_bonus * (1.0 + inner_closeness)
        if curr_in_dwell_region:
            dwell_closeness = max(1.0 - curr_pos_norm / dwell_r, 0.0)
            basin_dwell_region_bonus = cfg.basin_dwell_bonus * (1.0 + dwell_closeness)

        basin_outer_exit_penalty = -cfg.basin_outer_exit_penalty if prev_in_outer and not curr_in_outer else 0.0
        basin_inner_exit_penalty = -cfg.basin_inner_exit_penalty if prev_in_inner and not curr_in_inner else 0.0
        basin_dwell_break_penalty = -cfg.basin_dwell_break_penalty if prev_in_dwell_region and not curr_in_dwell_region else 0.0
        basin_drift_penalty = (
            -cfg.basin_drift_penalty_weight * max(curr_pos_norm - prev_pos_norm, 0.0)
            if (prev_in_outer or curr_in_outer)
            else 0.0
        )

    components = {
        "position_progress": float(position_progress),
        "orientation_progress": float(orientation_progress),
        "stay_in_zone_bonus": float(stay_in_zone),
        "dwell_bonus": float(dwell_bonus),
        "working_range_bonus": float(working_range_bonus),
        "working_range_dwell_bonus": float(working_range_dwell_bonus),
        "tight_pose_bonus": float(tight_pose_bonus),
        "tight_pose_dwell_bonus": float(tight_pose_dwell_bonus),
        "strict_pose_leave_penalty": float(strict_pose_leave_penalty),
        "strict_center_reward": float(strict_center_reward),
        "strict_center_position_penalty": float(strict_center_position_penalty),
        "strict_center_orientation_penalty": float(strict_center_orientation_penalty),
        "strict_center_small_action_bonus": float(strict_center_small_action_bonus),
        "strict_center_dwell_bonus": float(strict_center_dwell_bonus),
        "tight_position_shaping": float(tight_position_shaping),
        "tight_orientation_shaping": float(tight_orientation_shaping),
        "convergence_position_progress": float(convergence_position_progress),
        "convergence_orientation_progress": float(convergence_orientation_progress),
        "orientation_position_gate_scale": float(orientation_position_gate_scale),
        "entry_action_penalty_scale": float(entry_action_penalty_scale),
        "leave_zone_penalty": float(leave_zone_penalty),
        "working_range_exit_penalty": float(working_range_exit_penalty),
        "drift_penalty": float(drift_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "action_delta_violation_penalty": float(action_delta_violation_penalty),
        "delta_q_change_penalty": float(delta_q_change_penalty),
        "preserve_state_bonus": float(preserve_state_bonus),
        "strict_hold_bonus": float(strict_hold_bonus),
        "low_motion_bonus": float(low_motion_bonus),
        "tiny_correction_bonus": float(tiny_correction_bonus),
        "worse_than_entry_penalty": float(worse_than_entry_penalty),
        "near_strict_regression_penalty": float(near_strict_regression_penalty),
        "aggressive_action_penalty": float(aggressive_action_penalty),
        "dq_penalty": float(dq_penalty),
        "joint_limit_penalty": float(joint_limit_penalty),
        "success_bonus": float(success_bonus),
        "basin_outer_bonus": float(basin_outer_bonus),
        "basin_inner_bonus": float(basin_inner_bonus),
        "basin_dwell_bonus": float(basin_dwell_region_bonus),
        "basin_outer_exit_penalty": float(basin_outer_exit_penalty),
        "basin_inner_exit_penalty": float(basin_inner_exit_penalty),
        "basin_dwell_break_penalty": float(basin_dwell_break_penalty),
        "basin_drift_penalty": float(basin_drift_penalty),
        "basin_zone_index": float(basin_zone_index),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "dwell_count": float(dwell_count),
        "in_tight_pose": float(curr_in_tight_pose),
        "in_near_strict": float(curr_in_near_strict),
        "entry_pos_error": float(entry_pos),
        "entry_ori_error": float(entry_ori),
        "entry_action_l2": float(entry_action),
        "entry_dq_norm": float(entry_dq),
        "entry_to_curr_delta_position_error": float(entry_to_curr_delta_position_error),
        "entry_to_curr_delta_orientation_error": float(entry_to_curr_delta_orientation_error),
        "entry_to_curr_delta_action_l2": float(entry_to_curr_delta_action_l2),
        "entry_to_curr_delta_dq_norm": float(entry_to_curr_delta_dq_norm),
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
            "working_range_bonus",
            "working_range_dwell_bonus",
            "tight_pose_bonus",
            "tight_pose_dwell_bonus",
            "strict_pose_leave_penalty",
            "strict_center_reward",
            "strict_center_position_penalty",
            "strict_center_orientation_penalty",
            "strict_center_small_action_bonus",
            "strict_center_dwell_bonus",
            "tight_position_shaping",
            "tight_orientation_shaping",
            "convergence_position_progress",
            "convergence_orientation_progress",
            "leave_zone_penalty",
            "working_range_exit_penalty",
            "drift_penalty",
            "smoothness_penalty",
            "action_delta_violation_penalty",
            "delta_q_change_penalty",
            "preserve_state_bonus",
            "strict_hold_bonus",
            "low_motion_bonus",
            "tiny_correction_bonus",
            "worse_than_entry_penalty",
            "near_strict_regression_penalty",
            "aggressive_action_penalty",
            "dq_penalty",
            "joint_limit_penalty",
            "success_bonus",
            "basin_outer_bonus",
            "basin_inner_bonus",
            "basin_dwell_bonus",
            "basin_outer_exit_penalty",
            "basin_inner_exit_penalty",
            "basin_dwell_break_penalty",
            "basin_drift_penalty",
        )
    )
    return float(reward), components
