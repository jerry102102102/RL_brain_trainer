"""Feature extraction for dock-readiness classification."""

from __future__ import annotations

from typing import Any

import numpy as np

from .handoff_dataset import ACTION_BUCKETS, ORIENTATION_BUCKETS, POSITION_BUCKETS, bucketize


OBS_FEATURE_KEYS = (
    "q",
    "dq",
    "prev_action",
    "goal_pos_err",
    "goal_ori_err",
    "task_type",
    "mode_flag",
    "progress",
    "joint_limit_margin",
)


def flatten_observation(observation: dict[str, Any], *, keys: tuple[str, ...] = OBS_FEATURE_KEYS) -> list[float]:
    values: list[float] = []
    for key in keys:
        arr = np.asarray(observation.get(key, []), dtype=np.float32).reshape(-1)
        values.extend(float(x) for x in arr)
    return values


def feature_vector(record: dict[str, Any]) -> list[float]:
    features = flatten_observation(record.get("observation", {}))
    scalar_keys = (
        "position_error",
        "orientation_error",
        "dwell_count",
        "action_magnitude",
        "dq_norm",
        "regression_signal",
    )
    features.extend(float(record.get(key, 0.0)) for key in scalar_keys)
    return features


def annotate_handoff_record(
    *,
    observation: dict[str, Any],
    info: dict[str, Any],
    action_magnitude: float,
    rollout_id: int,
    episode_id: int,
    step: int,
    source_policy_checkpoint: str,
    switch_rule_config: dict[str, float | int],
) -> dict[str, Any]:
    position_error = float(info["position_error_norm"])
    orientation_error = float(info["orientation_error_norm"])
    dwell_count = int(info.get("dwell_count", 0))
    regression_signal = float(position_error - float(info.get("min_position_error", position_error)))
    dq = np.asarray(info.get("dq", []), dtype=float)
    is_pos_near = position_error <= float(switch_rule_config.get("dock_enter_pos_threshold_m", 0.009))
    is_ori_coarse = orientation_error <= float(switch_rule_config.get("dock_enter_ori_threshold_rad", 3.2))
    is_dwell_ready = dwell_count >= int(switch_rule_config.get("dock_enter_dwell_steps", 2))
    is_action_ready = action_magnitude <= float(switch_rule_config.get("dock_enter_action_threshold", 0.30))
    is_regression_ready = regression_signal <= float(switch_rule_config.get("dock_enter_regression_threshold_m", 0.003))
    is_switch_rule_ready = bool(is_pos_near and is_ori_coarse and is_dwell_ready and is_action_ready and is_regression_ready)
    return {
        "observation": observation,
        "q": np.asarray(info["q"], dtype=float).tolist(),
        "dq": dq.tolist(),
        "prev_action": np.asarray(observation.get("prev_action", []), dtype=float).tolist(),
        "goal_q": np.asarray(info.get("goal_q", []), dtype=float).tolist(),
        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
        "task_type": np.asarray(observation.get("task_type", []), dtype=float).tolist(),
        "mode_flag": np.asarray(observation.get("mode_flag", []), dtype=float).tolist(),
        "position_error": position_error,
        "orientation_error": orientation_error,
        "dwell_count": dwell_count,
        "action_magnitude": float(action_magnitude),
        "dq_norm": float(np.linalg.norm(dq)) if dq.size else 0.0,
        "regression_signal": regression_signal,
        "current_step": int(step),
        "source_episode_id": int(episode_id),
        "source_rollout_id": int(rollout_id),
        "source_policy_checkpoint": source_policy_checkpoint,
        "curriculum_stage_index": int(info.get("curriculum_stage_index", 0)),
        "curriculum_stage_name": str(info.get("curriculum_stage_name", "unknown")),
        "is_pos_near": bool(is_pos_near),
        "is_ori_coarse": bool(is_ori_coarse),
        "is_dwell_ready": bool(is_dwell_ready),
        "is_switch_rule_ready": bool(is_switch_rule_ready),
        "position_error_bucket": bucketize(position_error, POSITION_BUCKETS, unit="m"),
        "orientation_error_bucket": bucketize(orientation_error, ORIENTATION_BUCKETS, unit="rad"),
        "action_magnitude_bucket": bucketize(action_magnitude, ACTION_BUCKETS),
    }


__all__ = ["OBS_FEATURE_KEYS", "annotate_handoff_record", "feature_vector", "flatten_observation"]
