"""Rule-based switching between approach and dock policies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SwitchingConfig:
    dock_enter_pos_threshold_m: float = 0.08
    dock_enter_ori_threshold_rad: float = 0.25
    dock_enter_dwell_steps: int = 2
    dock_enter_action_threshold: float = 0.35
    dock_enter_regression_threshold_m: float = 0.01
    dock_enter_confirm_steps: int = 2
    dock_exit_pos_threshold_m: float = 0.10
    dock_exit_ori_threshold_rad: float = 0.35
    dock_exit_regression_threshold_m: float = 0.03
    dock_exit_confirm_steps: int = 2
    dock_timeout_steps: int = 8
    dock_timeout_min_improvement_m: float = 0.01
    min_approach_steps_before_switch: int = 3


def is_ready_to_dock(
    *,
    position_error_norm: float,
    orientation_error_norm: float,
    dwell_count: int,
    action_magnitude: float,
    min_position_error_so_far: float,
    config: SwitchingConfig,
) -> bool:
    if position_error_norm > config.dock_enter_pos_threshold_m:
        return False
    if orientation_error_norm > config.dock_enter_ori_threshold_rad:
        return False
    if dwell_count < config.dock_enter_dwell_steps:
        return False
    if action_magnitude > config.dock_enter_action_threshold:
        return False
    if position_error_norm - min_position_error_so_far > config.dock_enter_regression_threshold_m:
        return False
    return True


@dataclass
class TwoPolicySwitcher:
    config: SwitchingConfig = field(default_factory=SwitchingConfig)
    active_mode: str = "approach"
    switch_count: int = 0
    switch_steps: list[int] = field(default_factory=list)
    ready_to_dock_trigger_count: int = 0
    ready_to_dock_confirmed_count: int = 0
    dock_timeout_count: int = 0
    switch_back_count: int = 0
    first_switch_step: int | None = None
    _enter_ready_streak: int = 0
    _exit_ready_streak: int = 0
    _dock_steps: int = 0
    _dock_entry_position_error: float | None = None
    _dock_best_position_error: float = float("inf")

    def reset(self) -> None:
        self.active_mode = "approach"
        self.switch_count = 0
        self.switch_steps = []
        self.ready_to_dock_trigger_count = 0
        self.ready_to_dock_confirmed_count = 0
        self.dock_timeout_count = 0
        self.switch_back_count = 0
        self.first_switch_step = None
        self._enter_ready_streak = 0
        self._exit_ready_streak = 0
        self._dock_steps = 0
        self._dock_entry_position_error = None
        self._dock_best_position_error = float("inf")

    def update(
        self,
        *,
        position_error_norm: float,
        orientation_error_norm: float,
        dwell_count: int,
        action_magnitude: float,
        min_position_error_so_far: float,
        step_index: int,
    ) -> str:
        next_mode = self.active_mode
        timeout_exit = False
        if self.active_mode == "approach":
            ready = (
                step_index >= self.config.min_approach_steps_before_switch
                and is_ready_to_dock(
                    position_error_norm=position_error_norm,
                    orientation_error_norm=orientation_error_norm,
                    dwell_count=dwell_count,
                    action_magnitude=action_magnitude,
                    min_position_error_so_far=min_position_error_so_far,
                    config=self.config,
                )
            )
            if ready:
                self.ready_to_dock_trigger_count += 1
                self._enter_ready_streak += 1
            else:
                self._enter_ready_streak = 0

            if self._enter_ready_streak >= self.config.dock_enter_confirm_steps:
                next_mode = "dock"
                self.ready_to_dock_confirmed_count += 1
                self._dock_steps = 0
                self._dock_entry_position_error = float(position_error_norm)
                self._dock_best_position_error = float(position_error_norm)
                self._exit_ready_streak = 0
        else:
            self._dock_steps += 1
            self._dock_best_position_error = min(self._dock_best_position_error, float(position_error_norm))
            timeout_exit = (
                self._dock_steps >= self.config.dock_timeout_steps
                and self._dock_entry_position_error is not None
                and (self._dock_entry_position_error - self._dock_best_position_error) < self.config.dock_timeout_min_improvement_m
            )
            leave_zone = position_error_norm >= self.config.dock_exit_pos_threshold_m
            bad_orientation = orientation_error_norm >= self.config.dock_exit_ori_threshold_rad
            regressed = (position_error_norm - self._dock_best_position_error) > self.config.dock_exit_regression_threshold_m
            if leave_zone or bad_orientation or regressed or timeout_exit:
                self._exit_ready_streak += 1
            else:
                self._exit_ready_streak = 0

            if self._exit_ready_streak >= self.config.dock_exit_confirm_steps:
                next_mode = "approach"
                self.switch_back_count += 1
                if timeout_exit:
                    self.dock_timeout_count += 1
                self._enter_ready_streak = 0
                self._exit_ready_streak = 0

        if next_mode != self.active_mode:
            self.active_mode = next_mode
            self.switch_count += 1
            self.switch_steps.append(int(step_index))
            if self.first_switch_step is None and next_mode == "dock":
                self.first_switch_step = int(step_index)
        return self.active_mode
