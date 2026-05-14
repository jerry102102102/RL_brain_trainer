"""Workspace expansion gate and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkspaceGateConfig:
    retention_stage0_4_success: float = 0.95
    retention_stage5_success: float = 0.85
    retention_stage_thresholds: tuple[float, ...] = ()
    promotion_stage_success: float = 0.80
    promotion_ready_rate: float = 0.80
    max_mean_position_error_m: float = 0.020
    max_mean_orientation_error_rad: float = 0.15
    score_current_success_weight: float = 0.45
    score_current_ready_weight: float = 0.20
    score_retention_weight: float = 0.20
    score_error_weight: float = 0.15


def stage_passed(stage_metrics: dict[str, Any], cfg: WorkspaceGateConfig) -> bool:
    return bool(
        float(stage_metrics.get("success_rate", 0.0)) >= cfg.promotion_stage_success
        and float(stage_metrics.get("finisher_ready_hit_rate", 0.0)) >= cfg.promotion_ready_rate
        and float(stage_metrics.get("mean_final_position_error", 999.0)) <= cfg.max_mean_position_error_m
        and float(stage_metrics.get("mean_final_orientation_error", 999.0)) <= cfg.max_mean_orientation_error_rad
    )


def retention_ok(stage_metrics: dict[int, dict[str, Any]], cfg: WorkspaceGateConfig) -> bool:
    if cfg.retention_stage_thresholds:
        for idx, threshold in enumerate(cfg.retention_stage_thresholds):
            if idx not in stage_metrics:
                continue
            if float(stage_metrics.get(idx, {}).get("success_rate", 0.0)) < float(threshold):
                return False
        return True
    for idx in range(5):
        if float(stage_metrics.get(idx, {}).get("success_rate", 0.0)) < cfg.retention_stage0_4_success:
            return False
    return float(stage_metrics.get(5, {}).get("success_rate", 0.0)) >= cfg.retention_stage5_success


def highest_passed_stage(stage_metrics: dict[int, dict[str, Any]], cfg: WorkspaceGateConfig) -> int:
    best = -1
    for idx in sorted(stage_metrics):
        if stage_passed(stage_metrics[idx], cfg):
            best = idx
        else:
            if idx >= 6:
                break
    return best


def gated_score(stage_metrics: dict[int, dict[str, Any]], current_stage: int, cfg: WorkspaceGateConfig) -> dict[str, Any]:
    current = stage_metrics.get(current_stage, {})
    retention_values = [float(stage_metrics.get(i, {}).get("success_rate", 0.0)) for i in range(0, min(6, current_stage + 1))]
    retention = sum(retention_values) / len(retention_values) if retention_values else 0.0
    pos_err = float(current.get("mean_final_position_error", 1.0))
    ori_err = float(current.get("mean_final_orientation_error", 1.0))
    pos_score = max(0.0, 1.0 - pos_err / max(cfg.max_mean_position_error_m, 1e-6))
    ori_score = max(0.0, 1.0 - ori_err / max(cfg.max_mean_orientation_error_rad, 1e-6))
    error_score = 0.5 * (pos_score + ori_score)
    score = (
        float(current.get("success_rate", 0.0)) * cfg.score_current_success_weight
        + float(current.get("finisher_ready_hit_rate", 0.0)) * cfg.score_current_ready_weight
        + retention * cfg.score_retention_weight
        + error_score * cfg.score_error_weight
    )
    return {
        "score": float(score),
        "current_stage": int(current_stage),
        "retention_ok": retention_ok(stage_metrics, cfg),
        "highest_passed_stage": int(highest_passed_stage(stage_metrics, cfg)),
        "current_stage_success_rate": float(current.get("success_rate", 0.0)),
        "current_stage_ready_rate": float(current.get("finisher_ready_hit_rate", 0.0)),
        "retention_mean_success_rate": float(retention),
        "error_score": float(error_score),
    }


def gate_config_from_dict(payload: dict[str, Any] | None) -> WorkspaceGateConfig:
    data = dict(payload or {})
    if "retention_stage_thresholds" in data:
        data["retention_stage_thresholds"] = tuple(float(v) for v in data["retention_stage_thresholds"])
    fields = WorkspaceGateConfig.__dataclass_fields__
    return WorkspaceGateConfig(**{k: v for k, v in data.items() if k in fields})
