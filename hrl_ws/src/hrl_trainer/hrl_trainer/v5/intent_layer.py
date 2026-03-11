"""V5 WP1 L1 intent-layer schemas, validation, and slot resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


MOVE_PLATE_PATTERN = re.compile(r"^MOVE_PLATE\(\s*([^,\s][^,]*?)\s*,\s*([^)]+?)\s*\)$")
DEFAULT_SLOT_MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "v5_slot_map.yaml"

L2_FORBIDDEN_FIELDS = {
    "skill_mode",
    "ee_target_pose",
    "delta_pose",
    "gripper_cmd",
    "speed_profile_id",
    "guard",
}
L3_FORBIDDEN_FIELDS = {
    "joint_trajectory",
    "trajectory_points",
    "spline_points",
    "time_parameterized_trajectory",
    "execution_status",
    "intervention_log",
}


class IntentFailureCode(str, Enum):
    UNREACHABLE = "UNREACHABLE"
    MISSING_OBJECT = "MISSING_OBJECT"
    TASK_DISAMBIGUATION_REQUIRED = "TASK_DISAMBIGUATION_REQUIRED"


class IntentValidationError(ValueError):
    """Raised when an intent packet violates schema or layer boundaries."""


class IntentResolutionError(RuntimeError):
    """Raised when L1 cannot resolve task inputs into a valid intent."""

    def __init__(self, code: IntentFailureCode, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class PoseCandidate:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    score: float = 1.0
    approach_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    pregrasp_offset: float = 0.08
    pos_std: float = 0.0
    yaw_std: float = 0.0


@dataclass(frozen=True)
class IntentConstraints:
    clearance_m: float = 0.02
    speed_cap: str = "NORMAL"
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ReachabilityHint:
    ik_feasible: bool
    min_clearance_est: float
    preferred_approach: str = "top_down"


@dataclass(frozen=True)
class GraspHint:
    pregrasp_offset: float
    approach_axis: tuple[float, float, float]
    wrist_yaw_range: tuple[float, float] = (-3.14, 3.14)


@dataclass(frozen=True)
class IntentPacket:
    object_id: str
    source_slot: str
    target_slot: str
    pick_pose_candidates: list[PoseCandidate]
    place_pose_candidates: list[PoseCandidate]
    constraints: IntentConstraints
    reachability_hint: ReachabilityHint
    grasp_hint: GraspHint
    subtask_graph: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotRegion:
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class SlotPose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    region_world: SlotRegion
    approach_pose_candidates: tuple[SlotPose, ...]
    place_pose_candidates: tuple[SlotPose, ...]
    allowed_objects: tuple[str, ...]
    priority: int = 0


@dataclass(frozen=True)
class ResolvedMovePlate:
    source_slot: SlotSpec
    target_slot: SlotSpec
    object_id: str


def _as_float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _mapping_from_packet(packet: IntentPacket | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(packet, Mapping):
        return dict(packet)
    if is_dataclass(packet):
        return asdict(packet)
    raise TypeError(f"Unsupported intent packet type: {type(packet)!r}")


def _find_forbidden_fields(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    forbidden = L2_FORBIDDEN_FIELDS | L3_FORBIDDEN_FIELDS
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            if key_str in forbidden:
                hits.append(next_path)
            hits.extend(_find_forbidden_fields(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            hits.extend(_find_forbidden_fields(item, f"{path}[{idx}]"))
    return hits


def validate_intent_packet(packet: IntentPacket | Mapping[str, Any]) -> None:
    payload = _mapping_from_packet(packet)
    required = {
        "object_id",
        "source_slot",
        "target_slot",
        "pick_pose_candidates",
        "place_pose_candidates",
        "constraints",
        "reachability_hint",
        "grasp_hint",
        "subtask_graph",
    }
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise IntentValidationError(f"Missing required fields: {missing}")

    forbidden_hits = _find_forbidden_fields(payload)
    if forbidden_hits:
        raise IntentValidationError(
            "IntentPacket crosses L1 boundary with forbidden L2/L3 fields: " + ", ".join(sorted(forbidden_hits))
        )

    if not isinstance(payload["object_id"], str) or not payload["object_id"]:
        raise IntentValidationError("object_id must be a non-empty string")
    if not isinstance(payload["source_slot"], str) or not payload["source_slot"]:
        raise IntentValidationError("source_slot must be a non-empty string")
    if not isinstance(payload["target_slot"], str) or not payload["target_slot"]:
        raise IntentValidationError("target_slot must be a non-empty string")

    pick_candidates = payload["pick_pose_candidates"]
    place_candidates = payload["place_pose_candidates"]
    if not isinstance(pick_candidates, list) or not pick_candidates:
        raise IntentValidationError("pick_pose_candidates must be a non-empty list")
    if not isinstance(place_candidates, list) or not place_candidates:
        raise IntentValidationError("place_pose_candidates must be a non-empty list")

    for field_name in ("pick_pose_candidates", "place_pose_candidates"):
        for candidate in payload[field_name]:
            if not isinstance(candidate, Mapping):
                raise IntentValidationError(f"{field_name} entries must be mappings")
            if "xyz" not in candidate or "rpy" not in candidate:
                raise IntentValidationError(f"{field_name} entries must include xyz and rpy")
            _as_float3(candidate["xyz"], field_name=f"{field_name}.xyz")
            _as_float3(candidate["rpy"], field_name=f"{field_name}.rpy")

    reachability = payload["reachability_hint"]
    if not isinstance(reachability, Mapping):
        raise IntentValidationError("reachability_hint must be a mapping")
    if not isinstance(reachability.get("ik_feasible"), bool):
        raise IntentValidationError("reachability_hint.ik_feasible must be bool")


def parse_move_plate(command: str) -> tuple[str, str]:
    match = MOVE_PLATE_PATTERN.match(command.strip())
    if not match:
        raise ValueError(f"Invalid MOVE_PLATE command: {command!r}")
    return match.group(1).strip(), match.group(2).strip()


class SlotMap:
    """Slot resolver for MOVE_PLATE(source_slot, target_slot)."""

    def __init__(self, slots: Sequence[SlotSpec]):
        if not slots:
            raise ValueError("SlotMap requires at least one slot")
        self._slots = tuple(slots)
        self._by_slot_id = {slot.slot_id: slot for slot in self._slots}
        if len(self._by_slot_id) != len(self._slots):
            raise ValueError("slot_id values must be unique")

    @property
    def slots(self) -> tuple[SlotSpec, ...]:
        return self._slots

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SlotMap":
        with Path(path).open("r", encoding="utf-8") as fp:
            return cls.from_dict(yaml.safe_load(fp) or {})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SlotMap":
        slot_entries = payload.get("slots")
        if not isinstance(slot_entries, list):
            raise ValueError("SlotMap payload must include a slots list")

        slots: list[SlotSpec] = []
        for entry in slot_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("slots entries must be mappings")
            region = entry.get("region_world") or {}
            slot = SlotSpec(
                slot_id=str(entry["slot_id"]),
                region_world=SlotRegion(
                    center_xyz=_as_float3(region["center_xyz"], field_name="region_world.center_xyz"),
                    size_xyz=_as_float3(region["size_xyz"], field_name="region_world.size_xyz"),
                    yaw=float(region["yaw"]),
                ),
                approach_pose_candidates=tuple(
                    SlotPose(
                        xyz=_as_float3(pose["xyz"], field_name="approach_pose_candidates.xyz"),
                        rpy=_as_float3(pose["rpy"], field_name="approach_pose_candidates.rpy"),
                    )
                    for pose in entry.get("approach_pose_candidates", [])
                ),
                place_pose_candidates=tuple(
                    SlotPose(
                        xyz=_as_float3(pose["xyz"], field_name="place_pose_candidates.xyz"),
                        rpy=_as_float3(pose["rpy"], field_name="place_pose_candidates.rpy"),
                    )
                    for pose in entry.get("place_pose_candidates", [])
                ),
                allowed_objects=tuple(str(obj_id) for obj_id in entry.get("allowed_objects", [])),
                priority=int(entry.get("priority", 0)),
            )
            slots.append(slot)
        return cls(slots)

    def _match_slot_selector(self, selector: str) -> SlotSpec:
        if selector in self._by_slot_id:
            return self._by_slot_id[selector]
        prefix_matches = [slot for slot in self._slots if slot.slot_id.startswith(selector)]
        if len(prefix_matches) > 1:
            raise IntentResolutionError(
                IntentFailureCode.TASK_DISAMBIGUATION_REQUIRED,
                f"Ambiguous slot selector {selector!r}",
                {"selector": selector, "candidates": [slot.slot_id for slot in prefix_matches]},
            )
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        raise KeyError(f"Unknown slot selector: {selector}")

    def resolve_move_plate(self, source_selector: str, target_selector: str) -> ResolvedMovePlate:
        source_slot = self._match_slot_selector(source_selector)
        target_slot = self._match_slot_selector(target_selector)
        if source_slot.slot_id == target_slot.slot_id:
            raise IntentResolutionError(
                IntentFailureCode.UNREACHABLE,
                "Source and target slots are identical",
                {"slot_id": source_slot.slot_id},
            )

        common_objects = sorted(set(source_slot.allowed_objects).intersection(target_slot.allowed_objects))
        if len(common_objects) > 1:
            raise IntentResolutionError(
                IntentFailureCode.TASK_DISAMBIGUATION_REQUIRED,
                "Multiple candidate objects satisfy source/target slots",
                {
                    "source_slot": source_slot.slot_id,
                    "target_slot": target_slot.slot_id,
                    "candidate_object_ids": common_objects,
                },
            )
        if not common_objects:
            raise IntentResolutionError(
                IntentFailureCode.UNREACHABLE,
                "No object is allowed in both source and target slot",
                {"source_slot": source_slot.slot_id, "target_slot": target_slot.slot_id},
            )
        if not source_slot.approach_pose_candidates or not target_slot.place_pose_candidates:
            raise IntentResolutionError(
                IntentFailureCode.UNREACHABLE,
                "Missing approach/place pose candidates for resolved slots",
                {"source_slot": source_slot.slot_id, "target_slot": target_slot.slot_id},
            )

        return ResolvedMovePlate(source_slot=source_slot, target_slot=target_slot, object_id=common_objects[0])


def load_runtime_slot_map(path: str | Path | None = None) -> SlotMap:
    """Load slot map from runtime config path used by V5 WP1 pipeline."""
    slot_map_path = Path(path) if path is not None else DEFAULT_SLOT_MAP_PATH
    if not slot_map_path.exists():
        raise FileNotFoundError(f"Slot map config not found: {slot_map_path}")
    return SlotMap.from_yaml(slot_map_path)


def _best_object_pose(
    object_estimates: Sequence[Mapping[str, Any]],
    object_id: str,
    *,
    now_sec: float,
    min_confidence: float,
    max_staleness_sec: float,
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    for estimate in object_estimates:
        if str(estimate.get("object_id")) != object_id:
            continue
        confidence = float(estimate.get("confidence", 0.0))
        stamp_sec = float(estimate.get("stamp_sec", -1.0))
        if confidence < min_confidence:
            continue
        if now_sec - stamp_sec > max_staleness_sec:
            continue
        if best is None or confidence > float(best.get("confidence", 0.0)):
            best = estimate
    return best


def build_intent_packet(
    command: str,
    slot_map: SlotMap,
    object_estimates: Sequence[Mapping[str, Any]],
    *,
    now_sec: float,
    min_confidence: float = 0.5,
    max_staleness_sec: float = 0.5,
    constraints: IntentConstraints | None = None,
) -> IntentPacket:
    source_selector, target_selector = parse_move_plate(command)
    resolved = slot_map.resolve_move_plate(source_selector, target_selector)
    best_pose = _best_object_pose(
        object_estimates,
        resolved.object_id,
        now_sec=now_sec,
        min_confidence=min_confidence,
        max_staleness_sec=max_staleness_sec,
    )
    if best_pose is None:
        raise IntentResolutionError(
            IntentFailureCode.MISSING_OBJECT,
            "No fresh object estimate passed confidence/staleness gates",
            {
                "object_id": resolved.object_id,
                "min_confidence": min_confidence,
                "max_staleness_sec": max_staleness_sec,
            },
        )

    pos_std = float(best_pose.get("pos_std", 0.0))
    yaw_std = float(best_pose.get("yaw_std", 0.0))
    intent_constraints = constraints or IntentConstraints()

    pick_candidates = [
        PoseCandidate(xyz=pose.xyz, rpy=pose.rpy, pos_std=pos_std, yaw_std=yaw_std)
        for pose in resolved.source_slot.approach_pose_candidates
    ]
    place_candidates = [
        PoseCandidate(xyz=pose.xyz, rpy=pose.rpy, pos_std=pos_std, yaw_std=yaw_std)
        for pose in resolved.target_slot.place_pose_candidates
    ]

    packet = IntentPacket(
        object_id=resolved.object_id,
        source_slot=resolved.source_slot.slot_id,
        target_slot=resolved.target_slot.slot_id,
        pick_pose_candidates=pick_candidates,
        place_pose_candidates=place_candidates,
        constraints=intent_constraints,
        reachability_hint=ReachabilityHint(
            ik_feasible=True,
            min_clearance_est=float(intent_constraints.clearance_m),
            preferred_approach="top_down",
        ),
        grasp_hint=GraspHint(
            pregrasp_offset=float(pick_candidates[0].pregrasp_offset),
            approach_axis=pick_candidates[0].approach_axis,
        ),
        subtask_graph={
            "nodes": ["APPROACH", "GRASP", "LIFT", "TRANSFER", "PLACE", "RETREAT"],
            "recovery_edges": [
                {"from": "APPROACH", "on_fail": "RETREAT"},
                {"from": "GRASP", "on_fail": "APPROACH"},
                {"from": "PLACE", "on_fail": "TRANSFER"},
            ],
        },
    )
    validate_intent_packet(packet)
    return packet
