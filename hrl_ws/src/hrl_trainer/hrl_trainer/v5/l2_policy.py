"""L2 policy selector/wiring for runtime and rollout usage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .intent_layer import IntentPacket
from .rl_l2_v0 import RLL2V0, RLL2V0Config
from .rl_action import SkillCommand
from .rule_l2_v0 import RuleL2V0, RuleL2V0Config

L2_POLICY_RULE_V0 = "rule_l2_v0"
L2_POLICY_RL_L2 = "rl_l2"
POLICY_MODE_RULE = "rule"
POLICY_MODE_SAC = "sac"
ACTION_SCHEMA_V1 = "v1"
ACTION_SCHEMA_V2 = "v2"
ACTION_SCHEMAS = (ACTION_SCHEMA_V1, ACTION_SCHEMA_V2)


class L2PolicyAdapter:
    def __init__(
        self,
        policy_id: str = L2_POLICY_RULE_V0,
        *,
        terminal_skill: str = "RETREAT",
        action_schema: str = ACTION_SCHEMA_V2,
        policy_mode: str = POLICY_MODE_RULE,
        checkpoint: str | None = None,
    ):
        self.policy_id = policy_id
        self.policy_mode = str(policy_mode)
        self.checkpoint = checkpoint
        if self.policy_mode not in {POLICY_MODE_RULE, POLICY_MODE_SAC}:
            raise ValueError(f"Unsupported policy_mode: {self.policy_mode}")
        if self.policy_mode == POLICY_MODE_SAC and checkpoint:
            # Runtime adapter accepts checkpoint path to align v5 eval/train wiring.
            # Detailed SAC rollout execution lives in v5_1 eval/train modules.
            _ = Path(checkpoint)

        if action_schema not in ACTION_SCHEMAS:
            raise ValueError(f"Unsupported action_schema: {action_schema}. Expected one of: {ACTION_SCHEMAS}")
        if policy_id == L2_POLICY_RULE_V0:
            self._policy = RuleL2V0(RuleL2V0Config(terminal_skill=terminal_skill, action_schema=action_schema))
        elif policy_id == L2_POLICY_RL_L2:
            self._policy = RLL2V0(RLL2V0Config(terminal_skill=terminal_skill, action_schema=action_schema))
        else:
            raise ValueError(f"Unsupported L2 policy_id: {policy_id}")

    @property
    def flow(self) -> tuple[str, ...]:
        return self._policy.flow

    def plan_rollout(self, intent_packet: IntentPacket | Mapping[str, Any]) -> list[SkillCommand]:
        return self._policy.rollout(intent_packet)


def build_l2_rollout(
    intent_packet: IntentPacket | Mapping[str, Any],
    *,
    policy_id: str = L2_POLICY_RULE_V0,
    terminal_skill: str = "RETREAT",
    action_schema: str = ACTION_SCHEMA_V2,
    policy_mode: str = POLICY_MODE_RULE,
    checkpoint: str | None = None,
) -> list[SkillCommand]:
    """Runtime-friendly helper to select an L2 policy and produce SkillCommand rollout."""
    adapter = L2PolicyAdapter(
        policy_id=policy_id,
        terminal_skill=terminal_skill,
        action_schema=action_schema,
        policy_mode=policy_mode,
        checkpoint=checkpoint,
    )
    return adapter.plan_rollout(intent_packet)
