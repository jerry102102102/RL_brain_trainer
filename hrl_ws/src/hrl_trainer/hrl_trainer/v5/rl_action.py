"""V5 WP1.5 RL action schema validation and SkillCommand adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Mapping, Sequence


SKILL_MODES = {
    "APPROACH",
    "INSERT_SUPPORT",
    "LIFT_CARRY",
    "PLACE",
    "WITHDRAW",
    "RETREAT",
    # legacy modes kept for compatibility with existing tests/artifacts
    "GRASP",
    "LIFT",
    "TRANSFER",
}
GRIPPER_CMDS = {"OPEN", "CLOSE", "HOLD"}
V2_DEPRECATED_GRIPPER_CMDS = {"OPEN", "CLOSE"}
V2_LEGACY_GRIPPER_FIRST_SKILL_MODES = {"GRASP", "LIFT", "TRANSFER"}
SPEED_PROFILES = {"SLOW", "NORMAL"}
FRAGILITY_MODE_HINTS = {"DEFAULT", "CAUTIOUS", "ROBUST"}
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
class USlotParams:
    insert_depth: float
    lateral_alignment: float
    vertical_clearance: float
    entry_yaw: float


@dataclass(frozen=True)
class TimingParams:
    approach_speed_scale: float
    lift_profile_id: str
    contact_settle_time: float


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
class RLActionV2:
    schema_version: str
    skill_mode: str
    gripper_cmd: str
    speed_profile_id: str
    guard: GuardParams
    u_slot_params: USlotParams
    timing_params: TimingParams
    fragility_mode_hint: str | None = None
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
    u_slot_params: USlotParams | None = None
    timing_params: TimingParams | None = None
    fragility_mode_hint: str | None = None


def _as_float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise RLActionValidationError(f"{field_name} must contain exactly 3 values")
    out: list[float] = []
    for idx, raw in enumerate(values):
        try:
            out.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise RLActionValidationError(f"{field_name}[{idx}] must be a number") from exc
    return (out[0], out[1], out[2])


def _as_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RLActionValidationError(f"{field_name} must be a number") from exc


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
    max_tilt = _as_float(guard.get("max_tilt", -1.0), field_name="guard.max_tilt")
    if not 0.0 <= max_tilt <= (math.pi / 2.0):
        raise RLActionValidationError("guard.max_tilt must be in [0, pi/2]")
    min_clearance = _as_float(guard.get("min_clearance", -1.0), field_name="guard.min_clearance")
    if not 0.0 <= min_clearance <= 0.20:
        raise RLActionValidationError("guard.min_clearance must be in [0.0, 0.20]")


def _validate_u_slot_params(params: Mapping[str, Any]) -> None:
    required = {"insert_depth", "lateral_alignment", "vertical_clearance", "entry_yaw"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise RLActionValidationError(f"Missing required fields in u_slot_params: {missing}")

    insert_depth = _as_float(params["insert_depth"], field_name="u_slot_params.insert_depth")
    if not 0.0 <= insert_depth <= 0.20:
        raise RLActionValidationError("u_slot_params.insert_depth must be in [0.0, 0.20]")

    lateral_alignment = _as_float(params["lateral_alignment"], field_name="u_slot_params.lateral_alignment")
    if not -0.10 <= lateral_alignment <= 0.10:
        raise RLActionValidationError("u_slot_params.lateral_alignment must be in [-0.10, 0.10]")

    vertical_clearance = _as_float(params["vertical_clearance"], field_name="u_slot_params.vertical_clearance")
    if not 0.0 <= vertical_clearance <= 0.20:
        raise RLActionValidationError("u_slot_params.vertical_clearance must be in [0.0, 0.20]")

    entry_yaw = _as_float(params["entry_yaw"], field_name="u_slot_params.entry_yaw")
    if not -math.pi <= entry_yaw <= math.pi:
        raise RLActionValidationError("u_slot_params.entry_yaw must be in [-pi, pi]")


def _validate_timing_params(params: Mapping[str, Any]) -> None:
    required = {"approach_speed_scale", "lift_profile_id", "contact_settle_time"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise RLActionValidationError(f"Missing required fields in timing_params: {missing}")

    approach_speed_scale = _as_float(
        params["approach_speed_scale"], field_name="timing_params.approach_speed_scale"
    )
    if not 0.10 <= approach_speed_scale <= 2.00:
        raise RLActionValidationError("timing_params.approach_speed_scale must be in [0.10, 2.00]")

    lift_profile_id = params["lift_profile_id"]
    if not isinstance(lift_profile_id, str) or not lift_profile_id:
        raise RLActionValidationError("timing_params.lift_profile_id must be a non-empty string")

    contact_settle_time = _as_float(
        params["contact_settle_time"], field_name="timing_params.contact_settle_time"
    )
    if not 0.0 <= contact_settle_time <= 2.0:
        raise RLActionValidationError("timing_params.contact_settle_time must be in [0.0, 2.0]")


def _validate_exactly_one_pose(action: Mapping[str, Any]) -> None:
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


def _validate_common_action_fields(action: Mapping[str, Any]) -> None:
    forbidden_hits = _find_forbidden_fields(action)
    if forbidden_hits:
        raise RLActionValidationError(
            "RLAction crosses L2 boundary with forbidden L3 fields: " + ", ".join(sorted(forbidden_hits))
        )

    if action["skill_mode"] not in SKILL_MODES:
        raise RLActionValidationError(f"skill_mode must be one of: {sorted(SKILL_MODES)}")
    if action["gripper_cmd"] not in GRIPPER_CMDS:
        raise RLActionValidationError(f"gripper_cmd must be one of: {sorted(GRIPPER_CMDS)}")
    if action["speed_profile_id"] not in SPEED_PROFILES:
        raise RLActionValidationError(f"speed_profile_id must be one of: {sorted(SPEED_PROFILES)}")

    _validate_exactly_one_pose(action)

    guard = action["guard"]
    if not isinstance(guard, Mapping):
        raise RLActionValidationError("guard must be a mapping")
    _validate_guard(guard)


def _validate_v2_u_slot_first_policy(action: Mapping[str, Any]) -> None:
    skill_mode = action["skill_mode"]
    if skill_mode in V2_LEGACY_GRIPPER_FIRST_SKILL_MODES:
        raise RLActionValidationError(
            "v2 is U-slot-first; legacy gripper-first skill_mode is not allowed: "
            + f"{skill_mode}. Use U-slot skill modes (e.g. APPROACH/INSERT_SUPPORT/LIFT_CARRY/PLACE)."
        )

    gripper_cmd = action["gripper_cmd"]
    if gripper_cmd in V2_DEPRECATED_GRIPPER_CMDS:
        raise RLActionValidationError(
            "v2 gripper_cmd OPEN/CLOSE is deprecated compatibility only. "
            "Rejecting gripper-only/legacy intent; provide U-slot intent fields and use gripper_cmd='HOLD'."
        )


def validate_rl_action_v1(payload: RLActionV1 | Mapping[str, Any]) -> None:
    action = _mapping_from_payload(payload)
    required = {"schema_version", "skill_mode", "gripper_cmd", "speed_profile_id", "guard"}
    missing = sorted(required - set(action.keys()))
    if missing:
        raise RLActionValidationError(f"Missing required fields: {missing}")

    if action["schema_version"] != "v1":
        raise RLActionValidationError("schema_version must be 'v1'")

    _validate_common_action_fields(action)


def validate_rl_action_v2(payload: RLActionV2 | Mapping[str, Any]) -> None:
    action = _mapping_from_payload(payload)
    required = {
        "schema_version",
        "skill_mode",
        "gripper_cmd",
        "speed_profile_id",
        "guard",
        "u_slot_params",
        "timing_params",
    }
    missing = sorted(required - set(action.keys()))
    if missing:
        raise RLActionValidationError(f"Missing required fields: {missing}")

    if action["schema_version"] != "v2":
        raise RLActionValidationError("schema_version must be 'v2'")

    _validate_common_action_fields(action)
    _validate_v2_u_slot_first_policy(action)

    u_slot_params = action["u_slot_params"]
    if not isinstance(u_slot_params, Mapping):
        raise RLActionValidationError("u_slot_params must be a mapping")
    _validate_u_slot_params(u_slot_params)

    timing_params = action["timing_params"]
    if not isinstance(timing_params, Mapping):
        raise RLActionValidationError("timing_params must be a mapping")
    _validate_timing_params(timing_params)

    fragility_mode_hint = action.get("fragility_mode_hint")
    if fragility_mode_hint is not None and fragility_mode_hint not in FRAGILITY_MODE_HINTS:
        raise RLActionValidationError(
            f"fragility_mode_hint must be one of: {sorted(FRAGILITY_MODE_HINTS)}"
        )


def validate_rl_action(payload: RLActionV1 | RLActionV2 | Mapping[str, Any]) -> None:
    action = _mapping_from_payload(payload)
    schema_version = action.get("schema_version")
    if schema_version == "v1":
        validate_rl_action_v1(action)
        return
    if schema_version == "v2":
        validate_rl_action_v2(action)
        return
    raise RLActionValidationError("schema_version must be either 'v1' or 'v2'")


def _parse_pose(pose: Mapping[str, Any]) -> Pose6D:
    return Pose6D(
        xyz=_as_float3(pose.get("xyz", []), field_name="pose.xyz"),
        rpy=_as_float3(pose.get("rpy", []), field_name="pose.rpy"),
    )


def _parse_u_slot_params(params: Mapping[str, Any]) -> USlotParams:
    return USlotParams(
        insert_depth=float(params["insert_depth"]),
        lateral_alignment=float(params["lateral_alignment"]),
        vertical_clearance=float(params["vertical_clearance"]),
        entry_yaw=float(params["entry_yaw"]),
    )


def _parse_timing_params(params: Mapping[str, Any]) -> TimingParams:
    return TimingParams(
        approach_speed_scale=float(params["approach_speed_scale"]),
        lift_profile_id=str(params["lift_profile_id"]),
        contact_settle_time=float(params["contact_settle_time"]),
    )


def action_to_skill_command(payload: RLActionV1 | RLActionV2 | Mapping[str, Any]) -> SkillCommand:
    action = _mapping_from_payload(payload)
    validate_rl_action(action)

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
        u_slot_params=(
            _parse_u_slot_params(action["u_slot_params"]) if action.get("u_slot_params") is not None else None
        ),
        timing_params=(
            _parse_timing_params(action["timing_params"]) if action.get("timing_params") is not None else None
        ),
        fragility_mode_hint=action.get("fragility_mode_hint"),
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
