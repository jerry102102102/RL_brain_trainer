"""Route-specific reward for dense q-goal waypoint following."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.pose_utils import l2_norm, pose_error_components


@dataclass(frozen=True)
class RouteRewardConfig:
    q_goal_progress_weight: float = 2.0
    ee_position_progress_weight: float = 6.0
    ee_orientation_progress_weight: float = 5.0
    route_tangent_progress_weight: float = 0.25
    same_step_route_ready_bonus: float = 1.5
    route_ready_dwell_bonus: float = 0.8
    low_motion_near_waypoint_bonus: float = 0.4
    orientation_regression_penalty_weight: float = 4.0
    q_route_regression_penalty_weight: float = 1.0
    off_route_penalty_weight: float = 0.25
    action_magnitude_weight: float = 0.02
    action_delta_weight: float = 0.03
    dq_penalty_weight: float = 0.8
    no_progress_penalty: float = 0.02
    route_ready_pos_threshold_m: float = 0.010
    route_ready_ori_threshold_rad: float = 0.150
    route_ready_q_threshold: float = 0.080
    route_ready_action_threshold: float = 0.25
    route_ready_dq_threshold: float = 0.010


def route_ready(
    *,
    q_error_norm: float,
    pos_error_norm: float,
    ori_error_norm: float,
    action_norm: float,
    dq_norm: float,
    config: RouteRewardConfig,
) -> bool:
    return bool(
        q_error_norm <= config.route_ready_q_threshold
        and pos_error_norm <= config.route_ready_pos_threshold_m
        and ori_error_norm <= config.route_ready_ori_threshold_rad
        and action_norm <= config.route_ready_action_threshold
        and dq_norm <= config.route_ready_dq_threshold
    )


def compute_route_reward(
    *,
    prev_q: Sequence[float],
    curr_q: Sequence[float],
    goal_q: Sequence[float],
    prev_pose6: Sequence[float],
    curr_pose6: Sequence[float],
    goal_pose6: Sequence[float],
    route_tangent_q: Sequence[float],
    action: Sequence[float],
    prev_action: Sequence[float],
    prev_dq: Sequence[float],
    curr_dq: Sequence[float],
    ready_streak: int,
    nearest_route_q_distance: float = 0.0,
    config: RouteRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or RouteRewardConfig()
    prev_q_arr = np.asarray(prev_q, dtype=float)
    curr_q_arr = np.asarray(curr_q, dtype=float)
    goal_q_arr = np.asarray(goal_q, dtype=float)
    tangent = np.asarray(route_tangent_q, dtype=float)
    action_arr = np.asarray(action, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    curr_dq_arr = np.asarray(curr_dq, dtype=float)
    prev_dq_arr = np.asarray(prev_dq, dtype=float)

    prev_q_err = float(np.linalg.norm(goal_q_arr - prev_q_arr))
    curr_q_err = float(np.linalg.norm(goal_q_arr - curr_q_arr))
    prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, goal_pose6)
    curr_pos_err, curr_ori_err = pose_error_components(curr_pose6, goal_pose6)
    prev_pos_norm = l2_norm(prev_pos_err)
    curr_pos_norm = l2_norm(curr_pos_err)
    prev_ori_norm = l2_norm(prev_ori_err)
    curr_ori_norm = l2_norm(curr_ori_err)
    action_norm = float(np.linalg.norm(action_arr))
    dq_norm = float(np.linalg.norm(curr_dq_arr))
    tangent_norm = float(np.linalg.norm(tangent))
    q_delta = curr_q_arr - prev_q_arr
    tangent_progress = float(np.dot(q_delta, tangent) / max(tangent_norm, 1e-9)) if tangent_norm > 0.0 else 0.0
    ready_now = route_ready(
        q_error_norm=curr_q_err,
        pos_error_norm=curr_pos_norm,
        ori_error_norm=curr_ori_norm,
        action_norm=action_norm,
        dq_norm=dq_norm,
        config=cfg,
    )

    q_goal_progress = cfg.q_goal_progress_weight * (prev_q_err - curr_q_err)
    ee_position_progress = cfg.ee_position_progress_weight * (prev_pos_norm - curr_pos_norm)
    ee_orientation_progress = cfg.ee_orientation_progress_weight * (prev_ori_norm - curr_ori_norm)
    route_tangent_progress_bonus = cfg.route_tangent_progress_weight * max(tangent_progress, 0.0)
    same_step_route_ready_bonus = cfg.same_step_route_ready_bonus if ready_now else 0.0
    route_ready_dwell_bonus = cfg.route_ready_dwell_bonus if ready_now and ready_streak >= 1 else 0.0
    low_motion_near_waypoint_bonus = 0.0
    if curr_pos_norm <= 2.0 * cfg.route_ready_pos_threshold_m and curr_ori_norm <= 2.0 * cfg.route_ready_ori_threshold_rad:
        action_clean = max(1.0 - action_norm / max(cfg.route_ready_action_threshold, 1e-9), 0.0)
        dq_clean = max(1.0 - dq_norm / max(cfg.route_ready_dq_threshold, 1e-9), 0.0)
        low_motion_near_waypoint_bonus = cfg.low_motion_near_waypoint_bonus * 0.5 * (action_clean + dq_clean)

    orientation_regression_penalty = -cfg.orientation_regression_penalty_weight * max(curr_ori_norm - prev_ori_norm, 0.0)
    q_route_regression_penalty = -cfg.q_route_regression_penalty_weight * max(curr_q_err - prev_q_err, 0.0)
    off_route_penalty = -cfg.off_route_penalty_weight * float(max(nearest_route_q_distance, 0.0))
    action_smoothness_penalty = -cfg.action_magnitude_weight * float(np.mean(action_arr**2))
    action_smoothness_penalty += -cfg.action_delta_weight * float(np.mean((action_arr - prev_action_arr) ** 2))
    dq_penalty = -cfg.dq_penalty_weight * float(np.linalg.norm(curr_dq_arr))
    no_progress_penalty = -cfg.no_progress_penalty if (curr_q_err >= prev_q_err and curr_pos_norm >= prev_pos_norm and curr_ori_norm >= prev_ori_norm) else 0.0

    components = {
        "q_goal_progress": float(q_goal_progress),
        "ee_position_progress": float(ee_position_progress),
        "ee_orientation_progress": float(ee_orientation_progress),
        "route_tangent_progress_bonus": float(route_tangent_progress_bonus),
        "same_step_route_ready_bonus": float(same_step_route_ready_bonus),
        "route_ready_dwell_bonus": float(route_ready_dwell_bonus),
        "low_motion_near_waypoint_bonus": float(low_motion_near_waypoint_bonus),
        "orientation_regression_penalty": float(orientation_regression_penalty),
        "q_route_regression_penalty": float(q_route_regression_penalty),
        "off_route_penalty": float(off_route_penalty),
        "action_smoothness_penalty": float(action_smoothness_penalty),
        "dq_penalty": float(dq_penalty),
        "no_progress_penalty": float(no_progress_penalty),
        "curr_q_error": float(curr_q_err),
        "curr_pos_error": float(curr_pos_norm),
        "curr_ori_error": float(curr_ori_norm),
        "route_ready": float(ready_now),
    }
    reward = sum(v for k, v in components.items() if k not in {"curr_q_error", "curr_pos_error", "curr_ori_error", "route_ready"})
    return float(reward), components

