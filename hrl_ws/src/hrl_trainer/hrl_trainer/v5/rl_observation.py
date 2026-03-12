"""V5 WP1.5 RL observation v1 schema and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


STAGE_FLAGS = {
    "WORKSPACE_EXPLORE",
    "APPROACH",
    "GRASP",
    "LIFT",
    "TRANSFER",
    "PLACE",
    "RETREAT",
}
POLICY_HIDDEN_FIELDS = {
    "gt_object_pose",
    "tray_pose_gt",
    "joint_trajectory",
    "trajectory_points",
    "spline_points",
    "time_parameterized_trajectory",
}


class RLObservationValidationError(ValueError):
    """Raised when RL observation payload violates v1 schema."""


@dataclass(frozen=True)
class Pose6D:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class RobotState:
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    ee_pose: Pose6D
    gripper_opening: float


@dataclass(frozen=True)
class ObjectPoseEstimate:
    object_id: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    confidence: float
    pos_std: float
    yaw_std: float
    stamp_sec: float


@dataclass(frozen=True)
class RLObservationV1:
    schema_version: str
    obs_latent: tuple[float, ...]
    robot_state: RobotState
    stage_flag: str
    target_slot: str
    target_zone: str
    source_slot: str | None = None
    active_zone: str | None = None
    object_pose_est: ObjectPoseEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_float3(values: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise RLObservationValidationError(f"{field_name} must contain exactly 3 values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _as_float_tuple(values: Sequence[Any], *, field_name: str) -> tuple[float, ...]:
    if not values:
        raise RLObservationValidationError(f"{field_name} must be non-empty")
    return tuple(float(v) for v in values)


def _mapping_from_observation(payload: RLObservationV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if is_dataclass(payload):
        return asdict(payload)
    raise TypeError(f"Unsupported observation payload type: {type(payload)!r}")


def _find_forbidden_fields(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            if key_str in POLICY_HIDDEN_FIELDS:
                hits.append(next_path)
            hits.extend(_find_forbidden_fields(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            hits.extend(_find_forbidden_fields(item, f"{path}[{idx}]"))
    return hits


def validate_rl_observation_v1(payload: RLObservationV1 | Mapping[str, Any]) -> None:
    obs = _mapping_from_observation(payload)
    required = {
        "schema_version",
        "obs_latent",
        "robot_state",
        "stage_flag",
        "target_slot",
        "target_zone",
    }
    missing = sorted(required - set(obs.keys()))
    if missing:
        raise RLObservationValidationError(f"Missing required fields: {missing}")

    forbidden_hits = _find_forbidden_fields(obs)
    if forbidden_hits:
        raise RLObservationValidationError(
            "RLObservationV1 crosses policy boundary with hidden fields: " + ", ".join(sorted(forbidden_hits))
        )

    if obs["schema_version"] != "v1":
        raise RLObservationValidationError("schema_version must be 'v1'")

    obs_latent = obs["obs_latent"]
    if not isinstance(obs_latent, Sequence) or isinstance(obs_latent, (str, bytes)):
        raise RLObservationValidationError("obs_latent must be a numeric sequence")
    _as_float_tuple(obs_latent, field_name="obs_latent")

    if obs["stage_flag"] not in STAGE_FLAGS:
        raise RLObservationValidationError(f"stage_flag must be one of: {sorted(STAGE_FLAGS)}")

    for field_name in ("target_slot", "target_zone"):
        if not isinstance(obs[field_name], str) or not obs[field_name]:
            raise RLObservationValidationError(f"{field_name} must be a non-empty string")

    robot_state = obs["robot_state"]
    if not isinstance(robot_state, Mapping):
        raise RLObservationValidationError("robot_state must be a mapping")
    joint_positions = robot_state.get("joint_positions")
    joint_velocities = robot_state.get("joint_velocities")
    if not isinstance(joint_positions, Sequence) or isinstance(joint_positions, (str, bytes)) or not joint_positions:
        raise RLObservationValidationError("robot_state.joint_positions must be a non-empty sequence")
    if not isinstance(joint_velocities, Sequence) or isinstance(joint_velocities, (str, bytes)):
        raise RLObservationValidationError("robot_state.joint_velocities must be a sequence")
    if len(joint_positions) != len(joint_velocities):
        raise RLObservationValidationError("robot_state.joint_positions and joint_velocities must have same length")
    _as_float_tuple(joint_positions, field_name="robot_state.joint_positions")
    _as_float_tuple(joint_velocities, field_name="robot_state.joint_velocities")

    ee_pose = robot_state.get("ee_pose")
    if not isinstance(ee_pose, Mapping):
        raise RLObservationValidationError("robot_state.ee_pose must be a mapping")
    _as_float3(ee_pose.get("xyz", []), field_name="robot_state.ee_pose.xyz")
    _as_float3(ee_pose.get("rpy", []), field_name="robot_state.ee_pose.rpy")

    gripper_opening = robot_state.get("gripper_opening")
    if gripper_opening is None:
        raise RLObservationValidationError("robot_state.gripper_opening is required")
    gripper_opening_value = float(gripper_opening)
    if not 0.0 <= gripper_opening_value <= 1.0:
        raise RLObservationValidationError("robot_state.gripper_opening must be in [0, 1]")

    object_pose_est = obs.get("object_pose_est")
    if object_pose_est is not None:
        if not isinstance(object_pose_est, Mapping):
            raise RLObservationValidationError("object_pose_est must be a mapping when provided")
        if not isinstance(object_pose_est.get("object_id"), str) or not object_pose_est.get("object_id"):
            raise RLObservationValidationError("object_pose_est.object_id must be non-empty string")
        _as_float3(object_pose_est.get("xyz", []), field_name="object_pose_est.xyz")
        _as_float3(object_pose_est.get("rpy", []), field_name="object_pose_est.rpy")
        confidence = float(object_pose_est.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise RLObservationValidationError("object_pose_est.confidence must be in [0, 1]")
        if float(object_pose_est.get("pos_std", -1.0)) < 0.0:
            raise RLObservationValidationError("object_pose_est.pos_std must be >= 0")
        if float(object_pose_est.get("yaw_std", -1.0)) < 0.0:
            raise RLObservationValidationError("object_pose_est.yaw_std must be >= 0")
        float(object_pose_est.get("stamp_sec", 0.0))


def build_rl_observation_v1(
    *,
    obs_latent: Sequence[float],
    robot_state: RobotState,
    stage_flag: str,
    target_slot: str,
    target_zone: str,
    source_slot: str | None = None,
    active_zone: str | None = None,
    object_pose_est: ObjectPoseEstimate | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RLObservationV1:
    observation = RLObservationV1(
        schema_version="v1",
        obs_latent=tuple(float(v) for v in obs_latent),
        robot_state=robot_state,
        stage_flag=stage_flag,
        target_slot=target_slot,
        target_zone=target_zone,
        source_slot=source_slot,
        active_zone=active_zone,
        object_pose_est=object_pose_est,
        metadata=dict(metadata or {}),
    )
    validate_rl_observation_v1(observation)
    return observation
