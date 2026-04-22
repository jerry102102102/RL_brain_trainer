"""Observation builder for the Phase 1 kinematic environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..kinematics.joint_limits import JointSpec, joint_limit_margin, normalize_joint_deltas, normalize_joint_positions
from ..kinematics.pose_utils import normalize_vector, pose_error_components

TASK_TYPE_REACH = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class ObservationBuilderConfig:
    pos_err_scale_m: float = 0.5
    ori_err_scale_rad: float = math.pi


def mode_flag_vector(mode_index: int) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    vec[int(np.clip(mode_index, 0, 3))] = 1.0
    return vec


def build_observation(
    *,
    q: Sequence[float],
    dq: Sequence[float],
    prev_action: Sequence[float],
    current_pose6: Sequence[float],
    goal_pose6: Sequence[float],
    joint_specs: Sequence[JointSpec],
    episode_progress: float,
    dwell_progress: float,
    mode_index: int,
    current_waypoint_pose6: Sequence[float] | None = None,
    next_waypoint_pose6: Sequence[float] | None = None,
    config: ObservationBuilderConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or ObservationBuilderConfig()
    q_arr = np.asarray(q, dtype=float)
    dq_arr = np.asarray(dq, dtype=float)
    prev_action_arr = np.asarray(prev_action, dtype=float)
    current_pose = np.asarray(current_pose6, dtype=float)
    goal_pose = np.asarray(goal_pose6, dtype=float)

    goal_pos_err, goal_ori_err = pose_error_components(current_pose, goal_pose)

    wp_pose = np.asarray(current_waypoint_pose6, dtype=float) if current_waypoint_pose6 is not None else None
    next_wp_pose = np.asarray(next_waypoint_pose6, dtype=float) if next_waypoint_pose6 is not None else None

    if wp_pose is None:
        wp_pos_err = np.zeros(3, dtype=np.float32)
        wp_ori_err = np.zeros(3, dtype=np.float32)
    else:
        wp_pos, wp_ori = pose_error_components(current_pose, wp_pose)
        wp_pos_err = normalize_vector(wp_pos, cfg.pos_err_scale_m).astype(np.float32)
        wp_ori_err = normalize_vector(wp_ori, cfg.ori_err_scale_rad).astype(np.float32)

    if next_wp_pose is None:
        next_wp_pos_err = np.zeros(3, dtype=np.float32)
        next_wp_ori_err = np.zeros(3, dtype=np.float32)
    else:
        next_wp_pos, next_wp_ori = pose_error_components(current_pose, next_wp_pose)
        next_wp_pos_err = normalize_vector(next_wp_pos, cfg.pos_err_scale_m).astype(np.float32)
        next_wp_ori_err = normalize_vector(next_wp_ori, cfg.ori_err_scale_rad).astype(np.float32)

    observation = {
        "q": normalize_joint_positions(q_arr, joint_specs).astype(np.float32),
        "dq": normalize_joint_deltas(dq_arr, joint_specs).astype(np.float32),
        "prev_action": np.clip(prev_action_arr, -1.0, 1.0).astype(np.float32),
        "goal_pos_err": normalize_vector(goal_pos_err, cfg.pos_err_scale_m).astype(np.float32),
        "goal_ori_err": normalize_vector(goal_ori_err, cfg.ori_err_scale_rad).astype(np.float32),
        "wp_pos_err": wp_pos_err,
        "wp_ori_err": wp_ori_err,
        "next_wp_pos_err": next_wp_pos_err,
        "next_wp_ori_err": next_wp_ori_err,
        "task_type": TASK_TYPE_REACH.copy(),
        "mode_flag": mode_flag_vector(mode_index),
        "progress": np.array(
            [
                float(np.clip(episode_progress, 0.0, 1.0)),
                float(np.clip(dwell_progress, 0.0, 1.0)),
                0.0,
            ],
            dtype=np.float32,
        ),
        "joint_limit_margin": joint_limit_margin(q_arr, joint_specs).astype(np.float32),
    }
    return observation
