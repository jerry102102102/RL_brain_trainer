"""Qwen-facing MCP tool handlers for the V5 L1 bridge.

This module intentionally keeps Qwen at the L1 boundary.  The exposed tools let
an LLM/VLM inspect the task scene, resolve a structured intent packet, and
prepare a high-level Phase-1 skill request.  They do not expose raw joint
actions, trajectories, or executor-level controls.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .intent_layer import (
    IntentConstraints,
    IntentResolutionError,
    SlotMap,
    build_intent_packet,
    load_runtime_slot_map,
    validate_intent_packet,
)
from .runtime_model_registry import load_phase3a_model_registry

L1_ALLOWED_OUTPUTS = {
    "object_id",
    "source_slot",
    "target_slot",
    "constraints",
    "semantic_subtasks",
    "pick_pose_candidates",
    "place_pose_candidates",
    "reachability_hint",
    "grasp_hint",
    "subtask_graph",
}

FORBIDDEN_CONTROL_OUTPUTS = {
    "joint_trajectory",
    "trajectory_points",
    "spline_points",
    "time_parameterized_trajectory",
    "joint_action",
    "raw_action",
    "delta_q",
    "torque",
    "executor_command",
}


class QwenMcpToolError(ValueError):
    """Raised when a Qwen-facing tool request is invalid or unsafe."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _slot_to_summary(slot: Any) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "center_xyz": list(slot.region_world.center_xyz),
        "size_xyz": list(slot.region_world.size_xyz),
        "yaw": slot.region_world.yaw,
        "allowed_objects": list(slot.allowed_objects),
        "priority": slot.priority,
        "approach_pose_candidates": [_jsonable(pose) for pose in slot.approach_pose_candidates],
        "place_pose_candidates": [_jsonable(pose) for pose in slot.place_pose_candidates],
    }


def _default_object_estimates(slot_map: SlotMap, *, now_sec: float) -> list[dict[str, Any]]:
    object_ids = sorted({obj_id for slot in slot_map.slots for obj_id in slot.allowed_objects})
    return [
        {
            "object_id": object_id,
            "xyz": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "confidence": 0.99,
            "pos_std": 0.001,
            "yaw_std": 0.01,
            "stamp_sec": now_sec,
            "frame_id": "world",
            "source": "mcp_default_scene_proxy",
        }
        for object_id in object_ids
    ]


def _require_str(args: Mapping[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QwenMcpToolError(f"{key} must be a non-empty string")
    return value.strip()


def _safe_constraints(raw: Any) -> IntentConstraints:
    if raw is None:
        return IntentConstraints()
    if not isinstance(raw, Mapping):
        raise QwenMcpToolError("constraints must be an object")
    speed_cap = str(raw.get("speed_cap", "NORMAL")).upper()
    if speed_cap not in {"SLOW", "NORMAL"}:
        raise QwenMcpToolError("constraints.speed_cap must be SLOW or NORMAL")
    return IntentConstraints(
        clearance_m=float(raw.get("clearance_m", 0.02)),
        speed_cap=speed_cap,
        timeout_s=float(raw.get("timeout_s", 10.0)),
    )


def _safe_semantic_subtasks(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise QwenMcpToolError("semantic_subtasks must be an array when provided")
    forbidden = set(FORBIDDEN_CONTROL_OUTPUTS) | {"q_delta", "delta_q", "trajectory", "joint_targets"}
    subtasks: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise QwenMcpToolError(f"semantic_subtasks[{idx}] must be an object")
        illegal = sorted(set(str(key) for key in item.keys()) & forbidden)
        if illegal:
            raise QwenMcpToolError(f"semantic_subtasks[{idx}] contains forbidden control fields: {illegal}")
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        if not name or not description:
            raise QwenMcpToolError(f"semantic_subtasks[{idx}] requires name and description")
        subtasks.append(
            {
                "name": name,
                "description": description,
                "posture_constraint": str(item.get("posture_constraint", "")).strip(),
            }
        )
    return subtasks


def _find_forbidden_control_fields(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            if key_str in FORBIDDEN_CONTROL_OUTPUTS:
                hits.append(next_path)
            hits.extend(_find_forbidden_control_fields(value, next_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            hits.extend(_find_forbidden_control_fields(item, f"{path}[{idx}]"))
    return hits


class QwenMcpBridge:
    """Safe tool facade for letting Qwen call the V5 L1/L2 boundary."""

    def __init__(
        self,
        *,
        slot_map_path: str | Path | None = None,
        now_sec: float = 100.0,
        approach_checkpoint: str | None = None,
        finisher_checkpoint: str | None = None,
    ):
        self.slot_map_path = slot_map_path
        self.now_sec = float(now_sec)
        registry = load_phase3a_model_registry()
        self.approach_checkpoint = approach_checkpoint or registry.approach.checkpoint
        self.finisher_checkpoint = finisher_checkpoint or registry.finisher.checkpoint
        self._slot_map = load_runtime_slot_map(slot_map_path)
        self._tool_handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "get_l1_scene_context": self.get_l1_scene_context,
            "resolve_intent_packet": self.resolve_intent_packet,
            "prepare_phase1_skill_request": self.prepare_phase1_skill_request,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_l1_scene_context",
                "description": (
                    "Return the current L1 scene contract: known slots, allowed objects, "
                    "available high-level skills, and forbidden low-level control fields."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_slot_poses": {
                            "type": "boolean",
                            "description": "Include approach/place pose candidates for each slot.",
                            "default": True,
                        }
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "resolve_intent_packet",
                "description": (
                    "Resolve a structured L1 task proposal into a validated IntentPacket. "
                    "Use this after deciding object/source/target. This tool rejects L2/L3 controls."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string"},
                        "source_slot": {"type": "string"},
                        "target_slot": {"type": "string"},
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "clearance_m": {"type": "number"},
                                "speed_cap": {"type": "string", "enum": ["SLOW", "NORMAL"]},
                                "timeout_s": {"type": "number"},
                            },
                            "additionalProperties": False,
                        },
                        "object_estimates": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional perception estimates. Defaults to a scene proxy.",
                        },
                        "semantic_subtasks": {
                            "type": "array",
                            "description": "High-level L1 subtask plan. Must not contain q_delta, trajectories, or raw controls.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "posture_constraint": {"type": "string"},
                                },
                                "required": ["name", "description"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["source_slot", "target_slot"],
                    "additionalProperties": True,
                },
            },
            {
                "name": "prepare_phase1_skill_request",
                "description": (
                    "Prepare a dry-run high-level Approach->Finisher skill request from a validated "
                    "IntentPacket. This is a safe L2/L3 bridge description, not raw execution."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "intent_packet": {"type": "object"},
                        "dry_run": {
                            "type": "boolean",
                            "description": "Must remain true in this first integration version.",
                            "default": True,
                        },
                    },
                    "required": ["intent_packet"],
                    "additionalProperties": False,
                },
            },
        ]

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise QwenMcpToolError(f"Unknown tool: {name}")
        args = arguments or {}
        if not isinstance(args, Mapping):
            raise QwenMcpToolError("tool arguments must be an object")
        forbidden = _find_forbidden_control_fields(args)
        if forbidden:
            raise QwenMcpToolError("Request contains forbidden low-level control fields: " + ", ".join(forbidden))
        return handler(args)

    def get_l1_scene_context(self, args: Mapping[str, Any]) -> dict[str, Any]:
        include_slot_poses = bool(args.get("include_slot_poses", True))
        slots = []
        for slot in self._slot_map.slots:
            summary = _slot_to_summary(slot)
            if not include_slot_poses:
                summary.pop("approach_pose_candidates", None)
                summary.pop("place_pose_candidates", None)
            slots.append(summary)
        return {
            "schema_version": "v5.qwen_mcp.scene_context.v1",
            "l1_role": "semantic task understanding and intent generation",
            "l1_allowed_outputs": sorted(L1_ALLOWED_OUTPUTS),
            "forbidden_control_outputs": sorted(FORBIDDEN_CONTROL_OUTPUTS),
            "available_tools": [tool["name"] for tool in self.list_tools()],
            "available_high_level_pipeline": {
                "name": "phase1_approach_to_finisher",
                "description": "U-shaped arm kinematic skill stack: approach, preserve pose, and insert/finish.",
                "skills": ["APPROACH", "FINISHER"],
                "action_contract": "normalized joint delta inside L2/L3 only; Qwen must not emit it",
            },
            "slots": slots,
            "known_objects": sorted({obj for slot in self._slot_map.slots for obj in slot.allowed_objects}),
        }

    def resolve_intent_packet(self, args: Mapping[str, Any]) -> dict[str, Any]:
        source_slot = _require_str(args, "source_slot")
        target_slot = _require_str(args, "target_slot")
        object_id = str(args.get("object_id", "")).strip()
        if object_id:
            source = self._slot_map._match_slot_selector(source_slot)  # noqa: SLF001 - intentional resolver reuse
            target = self._slot_map._match_slot_selector(target_slot)  # noqa: SLF001 - intentional resolver reuse
            if object_id not in source.allowed_objects or object_id not in target.allowed_objects:
                raise QwenMcpToolError(
                    f"object_id {object_id!r} is not allowed by both source and target slots"
                )

        object_estimates = args.get("object_estimates")
        if object_estimates is None:
            object_estimates = _default_object_estimates(self._slot_map, now_sec=self.now_sec)
        if not isinstance(object_estimates, Sequence) or isinstance(object_estimates, (str, bytes)):
            raise QwenMcpToolError("object_estimates must be an array when provided")

        command = f"MOVE_PLATE({source_slot}, {target_slot})"
        try:
            semantic_subtasks = _safe_semantic_subtasks(args.get("semantic_subtasks"))
            packet = build_intent_packet(
                command,
                self._slot_map,
                object_estimates,  # type: ignore[arg-type]
                now_sec=self.now_sec,
                constraints=_safe_constraints(args.get("constraints")),
            )
        except IntentResolutionError as exc:
            raise QwenMcpToolError(
                json.dumps({"code": str(exc.code), "message": str(exc), "details": exc.details}, sort_keys=True)
            ) from exc
        validate_intent_packet(packet)
        return {
            "schema_version": "v5.qwen_mcp.intent_resolution.v1",
            "status": "ok",
            "command": command,
            "intent_packet": _jsonable(packet),
            "semantic_subtasks": semantic_subtasks,
            "next_recommended_tool": "prepare_phase1_skill_request",
        }

    def prepare_phase1_skill_request(self, args: Mapping[str, Any]) -> dict[str, Any]:
        dry_run = bool(args.get("dry_run", True))
        if not dry_run:
            raise QwenMcpToolError("This first MCP bridge only supports dry_run=true")

        intent_packet = args.get("intent_packet")
        if not isinstance(intent_packet, Mapping):
            raise QwenMcpToolError("intent_packet must be an object")
        validate_intent_packet(intent_packet)
        semantic_subtasks = _safe_semantic_subtasks(args.get("semantic_subtasks"))
        place_candidates = intent_packet.get("place_pose_candidates")
        if not isinstance(place_candidates, list) or not place_candidates:
            raise QwenMcpToolError("intent_packet.place_pose_candidates must be a non-empty list")
        target_pose = place_candidates[0]

        return {
            "schema_version": "v5.qwen_mcp.phase1_skill_request.v1",
            "status": "accepted_dry_run",
            "dry_run": True,
            "boundary_note": (
                "This request is intentionally high level. Qwen produced intent; "
                "L2/L3 own policy rollout, safety, and joint-level commands."
            ),
            "pipeline": "APPROACH -> FINISHER",
            "object_id": intent_packet["object_id"],
            "source_slot": intent_packet["source_slot"],
            "target_slot": intent_packet["target_slot"],
            "target_pose": target_pose,
            "semantic_subtasks": semantic_subtasks,
            "semantic_subtask_note": (
                "These are high-level L1/MCP semantic subtasks. They are not joint targets. "
                "The L2 runtime maps them into local RL target observations."
            ),
            "phase1_policy_assets": {
                "approach_checkpoint": self.approach_checkpoint,
                "finisher_checkpoint": self.finisher_checkpoint,
            },
            "constraints": intent_packet.get("constraints", {}),
        }
