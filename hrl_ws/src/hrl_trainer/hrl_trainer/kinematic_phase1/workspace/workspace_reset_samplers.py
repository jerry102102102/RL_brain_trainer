"""Workspace expansion reset design notes and config helpers.

The actual integrated stage mixing lives in ``envs.reset_samplers`` so the
existing ArmKinematicEnv can keep one reset path. This module centralizes the
intended ratios for documentation and future hard-case replay buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class WorkspaceStageMix:
    enabled: bool = True
    current_stage_ratio: float = 0.50
    previous_stage_ratio: float = 0.25
    old_workspace_replay_ratio: float = 0.20
    failure_replay_ratio: float = 0.05
    old_workspace_max_stage_index: int = 5

    def to_config(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_WORKSPACE_STAGE_MIX = WorkspaceStageMix()

