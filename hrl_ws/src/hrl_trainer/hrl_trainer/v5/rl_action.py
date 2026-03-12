"""V5 WP1.5 RL action v1 schema, validation, and SkillCommand adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Mapping, Sequence


SKILL_MODES = {"APPROACH", "GRASP", "LIFT", "TRANSFER", "PLACE", "RETREAT"}
GRIPPER_CMDS = {"OPEN", "CLOSE", "HOLD"}
SPEED_PROFILES = {"SLOW", "NORMAL"}
L3_FORBIDDEN_FIELDS = {
    "joint_trajectory",
    "trajectory_points",
    "spline_points",
    "time_parameterized_trajectory",
    "execution_status",
    "intervention_log",
}


class RLActionValidationError(ValueError):
    """Raised when RL action payload violates v1 schema or boundary rules."""


@dataclass(frozen=True)
class Pose6D:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class GuardParams:
    keep_level: bool
    max_tilt: float
    min_clearance: float


@dataclass(frozen=True)
class RLActionV1:
    schema_version: str
    skill_mode: str
    gripper_cmd: str
    speed_profile_id: str
    guard: GuardParams
    delta_pose: Pose6D | None = None
    ee_target_pose: Pose6D | None = None


@dataclass(frozen=True)
class SkillCommand:
    skill_mode: str
    gripper_cmd: str
    speed_profile_id: str
    guard: GuardParams
    delta_pose: Pose6D | None = None
    ee_target_pose: Pose6D | None = None


def _as_float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise RLActionValidationError(f"{field_name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _mapping_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if is_dataclass(payload):
        return asdict(payload)
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")


def _find_forbidden_fields(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            if key_str in L3_FORBIDDEN_FIELDS:
                hits.append(next_path)
            hits.extend(_find_forbidden_fields(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            hits.extend(_find_forbidden_fields(item, f"{path}[{idx}]"))
    return hits


def _validate_pose_bounds(field_name: str, pose: Mapping[str, Any]) -> None:
    xyz = _as_float3(pose.get("xyz", []), field_name=f"{field_name}.xyz")
    rpy = _as_float3(pose.get("rpy", []), field_name=f"{field_name}.rpy")
    if field_name == "delta_pose":
        for axis, value in zip(("x", "y", "z"), xyz):
            if abs(value) > 0.25:
                raise RLActionValidationError(f"delta_pose.{axis} must be in [-0.25, 0.25]")
        for axis, value in zip(("roll", "pitch", "yaw"), rpy):
            if abs(value) > (math.pi / 2.0):
                raise RLActionValidationError(f"delta_pose.{axis} must be in [-pi/2, pi/2]")


def _validate_guard(guard: Mapping[str, Any]) -> None:
    if not isinstance(guard.get("keep_level"), bool):
        raise RLActionValidationError("guard.keep_level must be bool")
    max_tilt = float(guard.get("max_tilt", -1.0))
    if not 0.0 <= max_tilt <= (math.pi / 2.0):
        raise RLActionValidationError("guard.max_tilt must be in [0, pi/2]")
    min_clearance = float(guard.get("min_clearance", -1.0))
    if not 0.0 <= min_clearance <= 0.20:
        raise RLActionValidationError("guard.min_clearance must be in [0.0, 0.20]")


def validate_rl_action_v1(payload: RLActionV1 | Mapping[str, Any]) -> None:
    action = _mapping_from_payload(payload)
    required = {"schema_version", "skill_mode", "gripper_cmd", "speed_profile_id", "guard"}
    missing = sorted(required - set(action.keys()))
    if missing:
        raise RLActionValidationError(f"Missing required fields: {missing}")

    if action["schema_version"] != "v1":
        raise RLActionValidationError("schema_version must be 'v1'")

    forbidden_hits = _find_forbidden_fields(action)
    if forbidden_hits:
        raise RLActionValidationError(
            "RLActionV1 crosses L2 boundary with forbidden L3 fields: " + ", ".join(sorted(forbidden_hits))
        )

    if action["skill_mode"] not in SKILL_MODES:
        raise RLActionValidationError(f"skill_mode must be one of: {sorted(SKILL_MODES)}")
    if action["gripper_cmd"] not in GRIPPER_CMDS:
        raise RLActionValidationError(f"gripper_cmd must be one of: {sorted(GRIPPER_CMDS)}")
    if action["speed_profile_id"] not in SPEED_PROFILES:
        raise RLActionValidationError(f"speed_profile_id must be one of: {sorted(SPEED_PROFILES)}")

    has_delta_pose = action.get("delta_pose") is not None
    has_ee_target_pose = action.get("ee_target_pose") is not None
    if has_delta_pose == has_ee_target_pose:
        raise RLActionValidationError("Exactly one of delta_pose or ee_target_pose must be provided")

    if has_delta_pose:
        if not isinstance(action["delta_pose"], Mapping):
            raise RLActionValidationError("delta_pose must be a mapping")
        _validate_pose_bounds("delta_pose", action["delta_pose"])
    if has_ee_target_pose:
        if not isinstance(action["ee_target_pose"], Mapping):
            raise RLActionValidationError("ee_target_pose must be a mapping")
        _validate_pose_bounds("ee_target_pose", action["ee_target_pose"])

    guard = action["guard"]
    if not isinstance(guard, Mapping):
        raise RLActionValidationError("guard must be a mapping")
    _validate_guard(guard)


def _parse_pose(pose: Mapping[str, Any]) -> Pose6D:
    return Pose6D(
        xyz=_as_float3(pose.get("xyz", []), field_name="pose.xyz"),
        rpy=_as_float3(pose.get("rpy", []), field_name="pose.rpy"),
    )


def action_to_skill_command(payload: RLActionV1 | Mapping[str, Any]) -> SkillCommand:
    action = _mapping_from_payload(payload)
    validate_rl_action_v1(action)

    guard = action["guard"]
    return SkillCommand(
        skill_mode=action["skill_mode"],
        gripper_cmd=action["gripper_cmd"],
        speed_profile_id=action["speed_profile_id"],
        guard=GuardParams(
            keep_level=bool(guard["keep_level"]),
            max_tilt=float(guard["max_tilt"]),
            min_clearance=float(guard["min_clearance"]),
        ),
        delta_pose=_parse_pose(action["delta_pose"]) if action.get("delta_pose") is not None else None,
        ee_target_pose=_parse_pose(action["ee_target_pose"]) if action.get("ee_target_pose") is not None else None,
    )


def validate_skill_command_boundary(payload: SkillCommand | Mapping[str, Any]) -> None:
    command = _mapping_from_payload(payload)
    forbidden_hits = _find_forbidden_fields(command)
    if forbidden_hits:
        raise RLActionValidationError(
            "SkillCommand crosses L2 boundary with forbidden L3 fields: " + ", ".join(sorted(forbidden_hits))
        )
    has_delta_pose = command.get("delta_pose") is not None
    has_ee_target_pose = command.get("ee_target_pose") is not None
    if has_delta_pose == has_ee_target_pose:
        raise RLActionValidationError("SkillCommand must carry exactly one of delta_pose or ee_target_pose")
