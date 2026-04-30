"""Draft three-stage Approach -> Bridge -> Dock switch state machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreeStageSwitchConfig:
    approach_to_bridge_pos_threshold_m: float = 0.030
    bridge_to_dock_readiness_threshold: float = 0.70
    bridge_to_dock_confirm_steps: int = 2
    dock_exit_pos_threshold_m: float = 0.030
    dock_exit_readiness_threshold: float = 0.20


@dataclass
class ThreeStageSwitcher:
    """Minimal state-machine sketch for future eval/training integration."""

    config: ThreeStageSwitchConfig
    active_mode: str = "approach"
    bridge_ready_streak: int = 0

    def reset(self) -> None:
        self.active_mode = "approach"
        self.bridge_ready_streak = 0

    def update(self, *, position_error: float, readiness_score: float | None = None) -> str:
        if self.active_mode == "approach" and position_error <= self.config.approach_to_bridge_pos_threshold_m:
            self.active_mode = "bridge"
            self.bridge_ready_streak = 0
        elif self.active_mode == "bridge":
            if readiness_score is not None and readiness_score >= self.config.bridge_to_dock_readiness_threshold:
                self.bridge_ready_streak += 1
            else:
                self.bridge_ready_streak = 0
            if self.bridge_ready_streak >= self.config.bridge_to_dock_confirm_steps:
                self.active_mode = "dock"
        elif self.active_mode == "dock":
            if position_error >= self.config.dock_exit_pos_threshold_m:
                self.active_mode = "bridge"
                self.bridge_ready_streak = 0
            if readiness_score is not None and readiness_score < self.config.dock_exit_readiness_threshold:
                self.active_mode = "bridge"
                self.bridge_ready_streak = 0
        return self.active_mode


__all__ = ["ThreeStageSwitchConfig", "ThreeStageSwitcher"]
