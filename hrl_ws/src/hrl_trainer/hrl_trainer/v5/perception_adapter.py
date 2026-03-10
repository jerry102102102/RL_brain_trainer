"""V5 WP1 perception adapter scaffolding for `/v5/perception/object_pose_est`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OBJECT_POSE_EST_TOPIC = "/v5/perception/object_pose_est"


class PerceptionAdapterError(ValueError):
    """Raised when perception adapter input/config is invalid."""


@dataclass(frozen=True)
class ObjectPoseEstimate:
    object_id: str
    xyz: tuple[float, float, float]
    yaw: float
    confidence: float
    pos_std: float
    yaw_std: float
    stamp_sec: float
    frame_id: str = "world"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "xyz": list(self.xyz),
            "yaw": self.yaw,
            "confidence": self.confidence,
            "pos_std": self.pos_std,
            "yaw_std": self.yaw_std,
            "stamp_sec": self.stamp_sec,
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class PerceptionAdapterConfig:
    mode: str = "phase0_gt_proxy"
    min_confidence: float = 0.5
    max_staleness_sec: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"phase0_gt_proxy", "phase1_vision_only"}:
            raise PerceptionAdapterError(f"Unsupported adapter mode: {self.mode}")
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise PerceptionAdapterError("min_confidence must be in [0, 1]")
        if self.max_staleness_sec < 0.0:
            raise PerceptionAdapterError("max_staleness_sec must be >= 0")


def _value(record: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise PerceptionAdapterError(f"{field_name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _coerce_record(record: Mapping[str, Any] | Any, *, default_confidence: float) -> ObjectPoseEstimate:
    xyz_raw = _value(record, "xyz")
    if xyz_raw is None and _value(record, "pose") is not None:
        xyz_raw = _value(_value(record, "pose"), "xyz")
    if xyz_raw is None:
        raise PerceptionAdapterError("Perception record missing xyz")
    return ObjectPoseEstimate(
        object_id=str(_value(record, "object_id")),
        xyz=_float3(xyz_raw, field_name="xyz"),
        yaw=float(_value(record, "yaw", 0.0)),
        confidence=float(_value(record, "confidence", default_confidence)),
        pos_std=float(_value(record, "pos_std", 0.0)),
        yaw_std=float(_value(record, "yaw_std", 0.0)),
        stamp_sec=float(_value(record, "stamp_sec", 0.0)),
        frame_id=str(_value(record, "frame_id", "world")),
    )


class PerceptionAdapter:
    """Adapts available estimators to the policy-visible object pose topic."""

    def __init__(self, config: PerceptionAdapterConfig):
        self.config = config

    @property
    def output_topic(self) -> str:
        return OBJECT_POSE_EST_TOPIC

    def adapt(
        self,
        *,
        gt_proxy_objects: Sequence[Mapping[str, Any] | Any] | None,
        vision_objects: Sequence[Mapping[str, Any] | Any] | None,
        now_sec: float | None = None,
    ) -> list[ObjectPoseEstimate]:
        if self.config.mode == "phase0_gt_proxy":
            if gt_proxy_objects is None:
                raise PerceptionAdapterError("phase0_gt_proxy mode requires gt_proxy_objects")
            source = gt_proxy_objects
            default_confidence = 1.0
        else:
            source = vision_objects or []
            default_confidence = 0.0

        by_object_id: dict[str, ObjectPoseEstimate] = {}
        for raw in source:
            estimate = _coerce_record(raw, default_confidence=default_confidence)
            if now_sec is not None and now_sec - estimate.stamp_sec > self.config.max_staleness_sec:
                continue
            if estimate.confidence < self.config.min_confidence:
                continue
            prev = by_object_id.get(estimate.object_id)
            if prev is None or estimate.confidence > prev.confidence:
                by_object_id[estimate.object_id] = estimate

        return [by_object_id[obj_id] for obj_id in sorted(by_object_id.keys())]
