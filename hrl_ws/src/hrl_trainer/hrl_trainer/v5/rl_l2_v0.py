"""Minimal RL-L2 v0 policy that emits SkillCommand v2 outputs only."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from .intent_layer import IntentPacket, validate_intent_packet
from .rl_action import GuardParams, SkillCommand, action_to_skill_command, validate_skill_command_boundary
from .rule_l2_v0 import BASE_FLOW, TERMINAL_FLOWS


@dataclass(frozen=True)
class RLL2V0Config:
    speed_profile_id: str = "NORMAL"
    guard: GuardParams = GuardParams(keep_level=True, max_tilt=0.30, min_clearance=0.03)
    terminal_skill: str = "WITHDRAW"
    action_schema: str = "v2"
    fragility_mode_hint: str = "CAUTIOUS"


class RLL2V0:
    """A simple heuristic policy used as a non-rule RL-L2 execution path."""

    def __init__(self, config: RLL2V0Config | None = None):
        self._config = config or RLL2V0Config()
        if self._config.terminal_skill not in TERMINAL_FLOWS:
            raise ValueError(f"terminal_skill must be one of: {TERMINAL_FLOWS}")
        if self._config.action_schema != "v2":
            raise ValueError("RLL2V0 currently supports action_schema='v2' only")
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

        mid_xy = ((pick_xyz[0] + place_xyz[0]) * 0.5, (pick_xyz[1] + place_xyz[1]) * 0.5)
        if stage == "APPROACH":
            target_xyz = (pick_xyz[0], pick_xyz[1], pick_xyz[2] + 0.10)
            target_rpy = [pick_rpy[0], pick_rpy[1], pick_rpy[2]]
            u_slot_params = self._u_slot_params(insert_depth=0.00, clearance=0.10, yaw=pick_rpy[2])
            timing_params = self._timing_params(speed=0.90, settle=0.10)
        elif stage == "INSERT_SUPPORT":
            target_xyz = (pick_xyz[0], pick_xyz[1], pick_xyz[2] + 0.005)
            target_rpy = [pick_rpy[0], pick_rpy[1], pick_rpy[2]]
            u_slot_params = self._u_slot_params(insert_depth=0.04, clearance=0.03, yaw=pick_rpy[2])
            timing_params = self._timing_params(speed=0.60, settle=0.30)
        elif stage == "LIFT_CARRY":
            target_xyz = (mid_xy[0], mid_xy[1], max(pick_xyz[2], place_xyz[2]) + 0.14)
            target_rpy = [pick_rpy[0], pick_rpy[1], pick_rpy[2]]
            u_slot_params = self._u_slot_params(insert_depth=0.03, clearance=0.14, yaw=pick_rpy[2])
            timing_params = self._timing_params(speed=1.10, settle=0.08)
        elif stage == "PLACE":
            target_xyz = (place_xyz[0], place_xyz[1], place_xyz[2] + 0.01)
            target_rpy = [place_rpy[0], place_rpy[1], place_rpy[2]]
            u_slot_params = self._u_slot_params(insert_depth=0.02, clearance=0.03, yaw=place_rpy[2])
            timing_params = self._timing_params(speed=0.80, settle=0.25)
        else:  # WITHDRAW / RETREAT
            target_xyz = (place_xyz[0], place_xyz[1], place_xyz[2] + 0.15)
            target_rpy = [place_rpy[0], place_rpy[1], place_rpy[2]]
            u_slot_params = self._u_slot_params(insert_depth=0.00, clearance=0.15, yaw=place_rpy[2])
            timing_params = self._timing_params(speed=1.00, settle=0.06)

        return {
            "schema_version": "v2",
            "skill_mode": stage,
            "ee_target_pose": {
                "xyz": [target_xyz[0], target_xyz[1], target_xyz[2]],
                "rpy": target_rpy,
            },
            "gripper_cmd": "HOLD",
            "speed_profile_id": self._config.speed_profile_id,
            "guard": {
                "keep_level": self._config.guard.keep_level,
                "max_tilt": self._config.guard.max_tilt,
                "min_clearance": self._config.guard.min_clearance,
            },
            "u_slot_params": u_slot_params,
            "timing_params": timing_params,
            "fragility_mode_hint": self._config.fragility_mode_hint,
        }

    def _u_slot_params(self, *, insert_depth: float, clearance: float, yaw: float) -> dict[str, float]:
        return {
            "insert_depth": float(insert_depth),
            "lateral_alignment": 0.0,
            "vertical_clearance": float(clearance),
            "entry_yaw": float(yaw),
        }

    def _timing_params(self, *, speed: float, settle: float) -> dict[str, Any]:
        return {
            "approach_speed_scale": float(speed),
            "lift_profile_id": "adaptive_lift_v0",
            "contact_settle_time": float(settle),
        }
