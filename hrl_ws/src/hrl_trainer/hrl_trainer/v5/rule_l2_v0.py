"""Rule-L2 v0 deterministic baseline for V5 U-slot execution flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from .intent_layer import IntentPacket, validate_intent_packet
from .rl_action import GuardParams, SkillCommand, action_to_skill_command, validate_skill_command_boundary

BASE_FLOW: tuple[str, ...] = ("APPROACH", "INSERT_SUPPORT", "LIFT_CARRY", "PLACE")
TERMINAL_FLOWS: tuple[str, ...] = ("WITHDRAW", "RETREAT")


@dataclass(frozen=True)
class RuleL2V0Config:
    speed_profile_id: str = "SLOW"
    guard: GuardParams = GuardParams(keep_level=True, max_tilt=0.40, min_clearance=0.02)
    terminal_skill: str = "RETREAT"
    action_schema: str = "v2"


class RuleL2V0:
    """Deterministic skill-sequence baseline that emits SkillCommand-only outputs."""

    def __init__(self, config: RuleL2V0Config | None = None):
        self._config = config or RuleL2V0Config()
        if self._config.terminal_skill not in TERMINAL_FLOWS:
            raise ValueError(f"terminal_skill must be one of: {TERMINAL_FLOWS}")
        if self._config.action_schema not in {"v1", "v2"}:
            raise ValueError("action_schema must be one of: ('v1', 'v2')")
        self._flow = BASE_FLOW + (self._config.terminal_skill,)

    @property
    def flow(self) -> tuple[str, ...]:
        return self._flow

    def rollout(self, intent_packet: IntentPacket | Mapping[str, Any]) -> list[SkillCommand]:
        validate_intent_packet(intent_packet)
        return [self.command_for_stage(intent_packet, stage_index=i) for i in range(len(self._flow))]

    def command_for_stage(self, intent_packet: IntentPacket | Mapping[str, Any], *, stage_index: int) -> SkillCommand:
        validate_intent_packet(intent_packet)
        if not 0 <= stage_index < len(self._flow):
            raise IndexError(f"stage_index must be in [0, {len(self._flow) - 1}]")

        packet = dict(intent_packet) if isinstance(intent_packet, Mapping) else asdict(intent_packet)
        stage = self._flow[stage_index]
        action = self._action_for_stage(stage, packet)
        command = action_to_skill_command(action)
        validate_skill_command_boundary(command)
        return command

    def _action_for_stage(self, stage: str, packet: Mapping[str, Any]) -> dict[str, Any]:
        pick_raw = packet["pick_pose_candidates"][0]
        place_raw = packet["place_pose_candidates"][0]
        pick = asdict(pick_raw) if is_dataclass(pick_raw) else dict(pick_raw)
        place = asdict(place_raw) if is_dataclass(place_raw) else dict(place_raw)

        pick_xyz = tuple(float(v) for v in pick["xyz"])
        pick_rpy = tuple(float(v) for v in pick["rpy"])
        place_xyz = tuple(float(v) for v in place["xyz"])
        place_rpy = tuple(float(v) for v in place["rpy"])

        if stage == "APPROACH":
            target_xyz = (pick_xyz[0], pick_xyz[1], pick_xyz[2] + 0.08)
            gripper_cmd = "OPEN"
        elif stage == "INSERT_SUPPORT":
            target_xyz = (pick_xyz[0], pick_xyz[1], pick_xyz[2] - 0.01)
            gripper_cmd = "OPEN"
        elif stage == "LIFT_CARRY":
            target_xyz = (pick_xyz[0], pick_xyz[1], pick_xyz[2] + 0.12)
            gripper_cmd = "CLOSE"
        elif stage == "PLACE":
            target_xyz = place_xyz
            gripper_cmd = "OPEN"
        else:  # WITHDRAW / RETREAT
            target_xyz = (place_xyz[0], place_xyz[1], place_xyz[2] + 0.12)
            gripper_cmd = "OPEN"

        target_rpy = (
            [place_rpy[0], place_rpy[1], place_rpy[2]]
            if stage in {"PLACE", "WITHDRAW", "RETREAT"}
            else [pick_rpy[0], pick_rpy[1], pick_rpy[2]]
        )
        guard = {
            "keep_level": self._config.guard.keep_level,
            "max_tilt": self._config.guard.max_tilt,
            "min_clearance": self._config.guard.min_clearance,
        }

        if self._config.action_schema == "v1":
            return {
                "schema_version": "v1",
                "skill_mode": stage,
                "ee_target_pose": {
                    "xyz": [target_xyz[0], target_xyz[1], target_xyz[2]],
                    "rpy": target_rpy,
                },
                "gripper_cmd": gripper_cmd,
                "speed_profile_id": self._config.speed_profile_id,
                "guard": guard,
            }

        u_slot_params = self._u_slot_params_for_stage(stage, pick_rpy=pick_rpy)
        timing_params = self._timing_params_for_stage(stage)
        return {
            "schema_version": "v2",
            "skill_mode": stage,
            "ee_target_pose": {
                "xyz": [target_xyz[0], target_xyz[1], target_xyz[2]],
                "rpy": target_rpy,
            },
            "gripper_cmd": "HOLD",
            "speed_profile_id": self._config.speed_profile_id,
            "guard": guard,
            "u_slot_params": u_slot_params,
            "timing_params": timing_params,
        }

    def _u_slot_params_for_stage(self, stage: str, *, pick_rpy: tuple[float, float, float]) -> dict[str, float]:
        entry_yaw = float(pick_rpy[2])
        if stage == "APPROACH":
            return {
                "insert_depth": 0.00,
                "lateral_alignment": 0.00,
                "vertical_clearance": 0.08,
                "entry_yaw": entry_yaw,
            }
        if stage == "INSERT_SUPPORT":
            return {
                "insert_depth": 0.03,
                "lateral_alignment": 0.00,
                "vertical_clearance": 0.02,
                "entry_yaw": entry_yaw,
            }
        if stage == "LIFT_CARRY":
            return {
                "insert_depth": 0.02,
                "lateral_alignment": 0.00,
                "vertical_clearance": 0.12,
                "entry_yaw": entry_yaw,
            }
        if stage == "PLACE":
            return {
                "insert_depth": 0.01,
                "lateral_alignment": 0.00,
                "vertical_clearance": 0.02,
                "entry_yaw": entry_yaw,
            }
        return {
            "insert_depth": 0.00,
            "lateral_alignment": 0.00,
            "vertical_clearance": 0.12,
            "entry_yaw": entry_yaw,
        }

    def _timing_params_for_stage(self, stage: str) -> dict[str, Any]:
        if stage == "INSERT_SUPPORT":
            approach_speed_scale = 0.70
            contact_settle_time = 0.25
        elif stage == "PLACE":
            approach_speed_scale = 0.80
            contact_settle_time = 0.20
        else:
            approach_speed_scale = 1.00
            contact_settle_time = 0.05
        return {
            "approach_speed_scale": approach_speed_scale,
            "lift_profile_id": "gentle_lift_v1",
            "contact_settle_time": contact_settle_time,
        }
