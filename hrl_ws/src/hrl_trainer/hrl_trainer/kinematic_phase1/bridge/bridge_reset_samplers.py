"""Bridge reset sampling from real Approach handoff datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..handoff.handoff_dataset import read_jsonl
from ..kinematics.joint_limits import JointSpec, clip_joint_configuration


@dataclass(frozen=True)
class BridgeResetConfig:
    handoff_dataset_path: str = ""
    max_position_error_m: float = 0.030
    max_orientation_error_rad: float = 4.0
    dirty_orientation_min_rad: float = 1.0
    dirty_motion_min_norm: float = 0.20
    dirty_orientation_probability: float = 0.70
    dirty_motion_probability: float = 0.10
    mixed_dirty_probability: float = 0.20
    perturb_q_noise: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _action_magnitude(record: dict) -> float:
    if "action_magnitude" in record:
        return float(record.get("action_magnitude", 0.0))
    return float(np.linalg.norm(np.asarray(record.get("prev_action", [0.0] * 7), dtype=float)))


def _dq_norm(record: dict) -> float:
    if "dq_norm" in record:
        return float(record.get("dq_norm", 0.0))
    return float(np.linalg.norm(np.asarray(record.get("dq", [0.0] * 7), dtype=float)))


def _bucket_type(record: dict, config: BridgeResetConfig) -> str:
    ori_dirty = float(record.get("orientation_error", 0.0)) >= config.dirty_orientation_min_rad
    motion_dirty = max(_dq_norm(record), _action_magnitude(record)) >= config.dirty_motion_min_norm
    if ori_dirty and motion_dirty:
        return "mixed_dirty_bucket"
    if ori_dirty:
        return "dirty_orientation_bucket"
    if motion_dirty:
        return "dirty_motion_bucket"
    return "near_goal_bucket"


def load_bridge_handoff_states(config: BridgeResetConfig) -> list[dict]:
    if not config.handoff_dataset_path:
        return []
    records = read_jsonl(Path(config.handoff_dataset_path))
    filtered: list[dict] = []
    for record in records:
        if float(record.get("position_error", 0.0)) > config.max_position_error_m:
            continue
        if float(record.get("orientation_error", 0.0)) > config.max_orientation_error_rad:
            continue
        next_record = dict(record)
        next_record["bridge_source_bucket_type"] = _bucket_type(record, config)
        next_record["bridge_source_dq_norm"] = _dq_norm(record)
        next_record["bridge_source_action_magnitude"] = _action_magnitude(record)
        filtered.append(next_record)
    return filtered


def _choose_bucket(rng: np.random.Generator, records: list[dict], config: BridgeResetConfig) -> str | None:
    available = {
        bucket
        for bucket in ("dirty_orientation_bucket", "dirty_motion_bucket", "mixed_dirty_bucket", "near_goal_bucket")
        if any(record.get("bridge_source_bucket_type") == bucket for record in records)
    }
    if not available:
        return None
    weighted = [
        ("dirty_orientation_bucket", max(config.dirty_orientation_probability, 0.0)),
        ("dirty_motion_bucket", max(config.dirty_motion_probability, 0.0)),
        ("mixed_dirty_bucket", max(config.mixed_dirty_probability, 0.0)),
    ]
    weighted = [(bucket, weight) for bucket, weight in weighted if bucket in available and weight > 0.0]
    if not weighted:
        return str(rng.choice(sorted(available)))
    total = sum(weight for _, weight in weighted)
    threshold = float(rng.random()) * total
    running = 0.0
    for bucket, weight in weighted:
        running += weight
        if threshold <= running:
            return bucket
    return weighted[-1][0]


def sample_bridge_reset(
    *,
    rng: np.random.Generator,
    records: list[dict],
    joint_specs: tuple[JointSpec, ...],
    config: BridgeResetConfig,
) -> dict:
    if not records:
        raise ValueError("Bridge reset requires a non-empty handoff dataset.")
    bucket = _choose_bucket(rng, records, config)
    candidates = [record for record in records if record.get("bridge_source_bucket_type") == bucket] if bucket else records
    if not candidates:
        candidates = records
    record = candidates[int(rng.integers(len(candidates)))]
    q = np.asarray(record["q"], dtype=float)
    noise = np.asarray(config.perturb_q_noise, dtype=float)
    if noise.size == q.size and np.any(noise > 0.0):
        q = clip_joint_configuration(q + rng.uniform(-noise, noise), joint_specs)
    return {
        "initial_q": q.tolist(),
        "initial_dq": record.get("dq", [0.0] * 7),
        "initial_prev_action": record.get("prev_action", [0.0] * 7),
        "goal_q": record.get("goal_q", [0.0] * 7),
        "goal_pose6": record["goal_pose6"],
        "source_bucket_type": record.get("bridge_source_bucket_type", "unknown"),
        "source_rollout_id": record.get("source_rollout_id"),
        "source_episode_id": record.get("source_episode_id"),
        "source_step": record.get("step"),
        "source_position_error": record.get("position_error"),
        "source_orientation_error": record.get("orientation_error"),
        "source_dwell_count": record.get("dwell_count"),
        "source_action_magnitude": record.get("bridge_source_action_magnitude", _action_magnitude(record)),
        "source_dq_norm": record.get("bridge_source_dq_norm", _dq_norm(record)),
    }


__all__ = ["BridgeResetConfig", "load_bridge_handoff_states", "sample_bridge_reset"]
