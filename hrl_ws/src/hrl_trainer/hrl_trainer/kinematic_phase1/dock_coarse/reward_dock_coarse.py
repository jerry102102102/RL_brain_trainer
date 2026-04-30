"""Retention-first Dock-Coarse reward.

Dock-Coarse is a basin expander: it should turn a coarse near-goal state into a
clean handoff state for the strict Dock-Finisher. It is deliberately not the
final 0.5 cm / 0.05 rad finisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class DockCoarseRewardConfig:
    # Dense cleanup shaping.
    position_progress_weight: float = 4.0
    orientation_progress_weight: float = 12.0
    dq_cleanup_weight: float = 0.35
    action_cleanup_weight: float = 0.20
    handoff_readiness_progress_weight: float = 3.0
    finisher_proxy_bonus_weight: float = 1.2
    handoff_motion_clean_bonus_weight: float = 0.45
    low_motion_handoff_bonus_weight: float = 1.0
    settle_bonus_weight: float = 0.8

    # Coarse and handoff-ready state definitions.
    coarse_pos_threshold_m: float = 0.010
    coarse_ori_threshold_rad: float = 0.20
    handoff_ready_pos_threshold_m: float = 0.010
    handoff_ready_ori_threshold_rad: float = 0.10
    handoff_ready_dq_threshold: float = 0.007
    handoff_ready_action_threshold: float = 0.45
    strict_like_pos_threshold_m: float = 0.005
    strict_like_ori_threshold_rad: float = 0.10
    working_pos_radius_m: float = 0.025
    working_ori_radius_rad: float = 0.35
    finisher_proxy_pos_scale_m: float = 0.010
    finisher_proxy_ori_scale_rad: float = 0.10
    finisher_proxy_dq_scale: float = 0.007
    finisher_proxy_action_scale: float = 0.45

    # State bonuses. Entry is intentionally small; dwell/readiness are primary.
    coarse_basin_bonus: float = 0.08
    coarse_retention_bonus: float = 0.16
    coarse_dwell_bonus: float = 0.70
    coarse_dwell_start: int = 2
    handoff_ready_bonus: float = 2.0
    handoff_ready_retention_bonus: float = 1.5
    handoff_ready_dwell_bonus: float = 1.0
    strict_like_bonus: float = 0.25
    working_range_bonus: float = 0.04
    coarse_success_bonus: float = 1.0

    # Retention and anti-regression.
    leave_working_range_penalty: float = 1.0
    leave_coarse_basin_penalty: float = 2.8
    leave_handoff_ready_penalty: float = 3.6
    leave_strict_like_penalty: float = 2.0
    position_regression_weight: float = 18.0
    orientation_regression_weight: float = 11.0
    working_range_regression_multiplier: float = 1.5
    coarse_basin_regression_multiplier: float = 3.0
    handoff_ready_regression_multiplier: float = 4.5

    # Near-basin smoothness. This gets heavier as the state becomes more useful.
    action_magnitude_weight: float = 0.06
    action_delta_weight: float = 0.12
    dq_norm_weight: float = 0.05
    working_range_smoothness_multiplier: float = 1.4
    coarse_basin_smoothness_multiplier: float = 2.8
    handoff_ready_smoothness_multiplier: float = 4.0

    joint_limit_penalty_weight: float = 0.05


def _bounded_closeness(value: float, scale: float) -> float:
    scale_f = max(float(scale), 1e-9)
    return float(np.exp(-((float(value) / scale_f) ** 2)))


def _finisher_proxy_score(
    *,
    pos: float,
    ori: float,
    dq_norm: float,
    action_norm: float,
    cfg: DockCoarseRewardConfig,
) -> float:
    """Cheap proxy for how comfortable the frozen strict finisher should be."""

    pose_score = 0.45 * _bounded_closeness(pos, cfg.finisher_proxy_pos_scale_m)
    pose_score += 0.35 * _bounded_closeness(ori, cfg.finisher_proxy_ori_scale_rad)
    motion_score = 0.10 * _bounded_closeness(dq_norm, cfg.finisher_proxy_dq_scale)
    motion_score += 0.10 * _bounded_closeness(action_norm, cfg.finisher_proxy_action_scale)
    return float(pose_score + motion_score)


def _zone_multiplier(
    *,
    prev_in_handoff_ready: bool,
    in_handoff_ready: bool,
    prev_in_coarse: bool,
    in_coarse: bool,
    prev_in_working: bool,
    in_working: bool,
    handoff_ready_value: float,
    coarse_value: float,
    working_value: float,
) -> float:
    if prev_in_handoff_ready or in_handoff_ready:
        return float(handoff_ready_value)
    if prev_in_coarse or in_coarse:
        return float(coarse_value)
    if prev_in_working or in_working:
        return float(working_value)
    return 1.0


def compute_dock_coarse_reward(
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
    dq_norm: float = 0.0,
    prev_dq_norm: float = 0.0,
    config: DockCoarseRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or DockCoarseRewardConfig()
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)
    prev_pos = l2_norm(prev_pos_err)
    curr_pos = l2_norm(curr_pos_err)
    prev_ori = l2_norm(prev_ori_err)
    curr_ori = l2_norm(curr_ori_err)
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    action_norm = float(np.linalg.norm(action_arr))
    prev_action_norm = float(np.linalg.norm(prev_action_arr))
    dq_norm_f = float(dq_norm)
    prev_dq_norm_f = float(prev_dq_norm)

    prev_in_working = prev_pos <= cfg.working_pos_radius_m and prev_ori <= cfg.working_ori_radius_rad
    in_working = curr_pos <= cfg.working_pos_radius_m and curr_ori <= cfg.working_ori_radius_rad
    prev_in_coarse = prev_pos <= cfg.coarse_pos_threshold_m and prev_ori <= cfg.coarse_ori_threshold_rad
    in_coarse = curr_pos <= cfg.coarse_pos_threshold_m and curr_ori <= cfg.coarse_ori_threshold_rad
    prev_in_handoff_ready = (
        prev_pos <= cfg.handoff_ready_pos_threshold_m
        and prev_ori <= cfg.handoff_ready_ori_threshold_rad
        and prev_dq_norm_f <= cfg.handoff_ready_dq_threshold
        and prev_action_norm <= cfg.handoff_ready_action_threshold
    )
    in_handoff_ready = (
        curr_pos <= cfg.handoff_ready_pos_threshold_m
        and curr_ori <= cfg.handoff_ready_ori_threshold_rad
        and dq_norm_f <= cfg.handoff_ready_dq_threshold
        and action_norm <= cfg.handoff_ready_action_threshold
    )
    prev_in_strict_like = prev_pos <= cfg.strict_like_pos_threshold_m and prev_ori <= cfg.strict_like_ori_threshold_rad
    in_strict_like = curr_pos <= cfg.strict_like_pos_threshold_m and curr_ori <= cfg.strict_like_ori_threshold_rad
    prev_finisher_proxy = _finisher_proxy_score(
        pos=prev_pos,
        ori=prev_ori,
        dq_norm=prev_dq_norm_f,
        action_norm=prev_action_norm,
        cfg=cfg,
    )
    curr_finisher_proxy = _finisher_proxy_score(
        pos=curr_pos,
        ori=curr_ori,
        dq_norm=dq_norm_f,
        action_norm=action_norm,
        cfg=cfg,
    )

    regression_multiplier = _zone_multiplier(
        prev_in_handoff_ready=prev_in_handoff_ready,
        in_handoff_ready=in_handoff_ready,
        prev_in_coarse=prev_in_coarse,
        in_coarse=in_coarse,
        prev_in_working=prev_in_working,
        in_working=in_working,
        handoff_ready_value=cfg.handoff_ready_regression_multiplier,
        coarse_value=cfg.coarse_basin_regression_multiplier,
        working_value=cfg.working_range_regression_multiplier,
    )
    smoothness_multiplier = _zone_multiplier(
        prev_in_handoff_ready=prev_in_handoff_ready,
        in_handoff_ready=in_handoff_ready,
        prev_in_coarse=prev_in_coarse,
        in_coarse=in_coarse,
        prev_in_working=prev_in_working,
        in_working=in_working,
        handoff_ready_value=cfg.handoff_ready_smoothness_multiplier,
        coarse_value=cfg.coarse_basin_smoothness_multiplier,
        working_value=cfg.working_range_smoothness_multiplier,
    )

    position_regression = max(curr_pos - prev_pos, 0.0)
    orientation_regression = max(curr_ori - prev_ori, 0.0)
    action_delta_mean_sq = float(np.mean((action_arr - prev_action_arr) ** 2))
    handoff_motion_clean_bonus = 0.0
    if in_coarse or in_handoff_ready:
        action_clean = max(1.0 - action_norm / max(cfg.finisher_proxy_action_scale, 1e-9), 0.0)
        dq_clean = max(1.0 - dq_norm_f / max(cfg.finisher_proxy_dq_scale, 1e-9), 0.0)
        handoff_motion_clean_bonus = cfg.handoff_motion_clean_bonus_weight * (0.5 * action_clean + 0.5 * dq_clean)
    low_motion_handoff_bonus = 0.0
    if in_handoff_ready:
        prev_action_clean = max(1.0 - prev_action_norm / max(cfg.finisher_proxy_action_scale, 1e-9), 0.0)
        action_clean = max(1.0 - action_norm / max(cfg.finisher_proxy_action_scale, 1e-9), 0.0)
        dq_clean = max(1.0 - dq_norm_f / max(cfg.finisher_proxy_dq_scale, 1e-9), 0.0)
        low_motion_handoff_bonus = cfg.low_motion_handoff_bonus_weight * (
            0.4 * action_clean + 0.3 * prev_action_clean + 0.3 * dq_clean
        )
    settle_progress = 0.0
    if in_coarse or in_handoff_ready or curr_finisher_proxy >= 0.55:
        settle_progress = 0.5 * max(prev_action_norm - action_norm, 0.0) + 0.5 * max(prev_dq_norm_f - dq_norm_f, 0.0)

    components = {
        "position_progress": cfg.position_progress_weight * (prev_pos - curr_pos),
        "orientation_progress": cfg.orientation_progress_weight * (prev_ori - curr_ori),
        "dq_cleanup_progress": cfg.dq_cleanup_weight * max(prev_dq_norm_f - dq_norm_f, 0.0),
        "action_cleanup_progress": cfg.action_cleanup_weight * max(prev_action_norm - action_norm, 0.0),
        "handoff_readiness_progress": cfg.handoff_readiness_progress_weight
        * max(curr_finisher_proxy - prev_finisher_proxy, 0.0),
        "finisher_proxy_bonus": cfg.finisher_proxy_bonus_weight * curr_finisher_proxy,
        "handoff_motion_clean_bonus": handoff_motion_clean_bonus,
        "low_motion_handoff_bonus": low_motion_handoff_bonus,
        "settle_bonus": cfg.settle_bonus_weight * settle_progress,
        "coarse_basin_bonus": cfg.coarse_basin_bonus if in_coarse else 0.0,
        "coarse_retention_bonus": cfg.coarse_retention_bonus if prev_in_coarse and in_coarse else 0.0,
        "coarse_dwell_bonus": (
            cfg.coarse_dwell_bonus * max(dwell_count - cfg.coarse_dwell_start + 1, 0)
            if in_coarse and dwell_count >= cfg.coarse_dwell_start
            else 0.0
        ),
        "handoff_ready_bonus": cfg.handoff_ready_bonus if in_handoff_ready else 0.0,
        "handoff_ready_retention_bonus": cfg.handoff_ready_retention_bonus if prev_in_handoff_ready and in_handoff_ready else 0.0,
        "handoff_ready_dwell_bonus": (
            cfg.handoff_ready_dwell_bonus * max(dwell_count - cfg.coarse_dwell_start + 1, 0)
            if in_handoff_ready and dwell_count >= cfg.coarse_dwell_start
            else 0.0
        ),
        "strict_like_bonus": cfg.strict_like_bonus if in_strict_like else 0.0,
        "working_range_bonus": cfg.working_range_bonus if in_working else 0.0,
        "leave_working_range_penalty": -cfg.leave_working_range_penalty if prev_in_working and not in_working else 0.0,
        "leave_coarse_basin_penalty": -cfg.leave_coarse_basin_penalty if prev_in_coarse and not in_coarse else 0.0,
        "leave_handoff_ready_penalty": -cfg.leave_handoff_ready_penalty if prev_in_handoff_ready and not in_handoff_ready else 0.0,
        "leave_strict_like_penalty": -cfg.leave_strict_like_penalty if prev_in_strict_like and not in_strict_like else 0.0,
        "position_regression_penalty": -cfg.position_regression_weight * regression_multiplier * position_regression,
        "orientation_regression_penalty": -cfg.orientation_regression_weight * regression_multiplier * orientation_regression,
        "near_basin_smoothness_penalty": smoothness_multiplier
        * (
            -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
            -cfg.action_delta_weight * action_delta_mean_sq
            -cfg.dq_norm_weight * dq_norm_f
        ),
        "joint_limit_penalty": -cfg.joint_limit_penalty_weight * float(max(0.25 - joint_limit_margin_min, 0.0) / 0.25),
        "coarse_success_bonus": cfg.coarse_success_bonus if success else 0.0,
        "curr_pos_error": curr_pos,
        "curr_ori_error": curr_ori,
        "curr_dq_norm": dq_norm_f,
        "curr_action_norm": action_norm,
        "prev_finisher_proxy_score": prev_finisher_proxy,
        "curr_finisher_proxy_score": curr_finisher_proxy,
        "regression_multiplier": regression_multiplier,
        "smoothness_multiplier": smoothness_multiplier,
        "in_working_range": float(in_working),
        "in_coarse_basin": float(in_coarse),
        "in_handoff_ready": float(in_handoff_ready),
        "in_strict_like_basin": float(in_strict_like),
        "left_working_range": float(prev_in_working and not in_working),
        "left_coarse_basin": float(prev_in_coarse and not in_coarse),
        "left_handoff_ready": float(prev_in_handoff_ready and not in_handoff_ready),
        "position_regression": float(position_regression > 0.0),
        "orientation_regression": float(orientation_regression > 0.0),
    }
    reward_terms = (
        "position_progress",
        "orientation_progress",
        "dq_cleanup_progress",
        "action_cleanup_progress",
        "handoff_readiness_progress",
        "finisher_proxy_bonus",
        "handoff_motion_clean_bonus",
        "low_motion_handoff_bonus",
        "settle_bonus",
        "coarse_basin_bonus",
        "coarse_retention_bonus",
        "coarse_dwell_bonus",
        "handoff_ready_bonus",
        "handoff_ready_retention_bonus",
        "handoff_ready_dwell_bonus",
        "strict_like_bonus",
        "working_range_bonus",
        "leave_working_range_penalty",
        "leave_coarse_basin_penalty",
        "leave_handoff_ready_penalty",
        "leave_strict_like_penalty",
        "position_regression_penalty",
        "orientation_regression_penalty",
        "near_basin_smoothness_penalty",
        "joint_limit_penalty",
        "coarse_success_bonus",
    )
    return float(sum(components[name] for name in reward_terms)), components


__all__ = ["DockCoarseRewardConfig", "compute_dock_coarse_reward"]
