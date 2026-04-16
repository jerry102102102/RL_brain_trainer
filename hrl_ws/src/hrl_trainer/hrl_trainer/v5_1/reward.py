"""Reward composition for V5.1 real SAC pipeline (phase-1 position-first ablation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    # Phase-1: position-first progress shaping.
    w_pos_progress_lin_toward: float = 6.0
    w_pos_progress_lin_away: float = 9.0
    w_pos_progress_away_near_scale: float = 3.0
    w_pos_progress_log: float = 0.12
    pos_log_eps: float = 1e-3

    # Orientation progress disabled for phase-1.
    w_ori_progress: float = 0.0

    # Mild smoothness regularization (squared L2).
    w_action_norm: float = -0.002
    w_jerk: float = -0.001
    w_adjust: float = 0.05
    w_raw_action: float = 0.01
    action_scale: float = 0.05

    # Keep safety penalties but lighter.
    w_intervention: float = -0.10
    w_clamp_projection: float = -0.12

    # Terminal rewards/penalties.
    timeout_penalty: float = -0.2
    reset_fail_penalty: float = -1.5
    success_bonus: float = 3.0
    execution_fail_penalty: float = -2.0
    reject_penalty: float = -0.5
    reject_delta_threshold: float = 0.8

    # Phase-1: disable these penalties.
    stall_penalty: float = 0.0
    ee_small_motion_penalty: float = 0.0

    # Hierarchical basin shaping.
    outer_shell_pos_m: float = 0.08
    inner_shell_pos_m: float = 0.04
    dwell_pos_m: float = 0.025
    near_goal_pos_m: float = 0.04  # compatibility alias for inner shell radius
    near_goal_shell_pos_m: float = 0.08  # compatibility alias for outer shell radius
    near_goal_bonus: float = 0.03  # entry bonus on first inner-shell entry
    shell_bonus: float = 0.05
    inner_shell_bonus: float = 0.10
    near_goal_shell_bonus: float = 0.05  # compatibility alias for shell bonus
    smooth_basin_enabled: bool = False
    smooth_basin_temperature_m: float = 0.015
    outer_exit_penalty: float = -0.10
    inner_exit_penalty: float = -0.20
    near_goal_exit_penalty: float = -0.20  # compatibility alias for inner exit
    drift_lambda: float = 8.0
    dwell_ori_rad: float = 0.12
    dwell_bonus: float = 0.30
    success_dwell_steps: int = 3
    dwell_steps_required: int = 3
    dwell_break_penalty: float = -0.30


@dataclass(frozen=True)
class RewardTerms:
    progress: float
    action: float
    jerk: float
    adjust_penalty: float
    raw_action_penalty: float
    reject_penalty: float
    intervention: float
    clamp_or_projection: float
    stall: float
    ee_small_motion_penalty: float
    timeout_penalty: float
    reset_fail_penalty: float
    execution_fail_penalty: float
    timeout_or_reset: float
    success_bonus: float
    near_goal: float
    near_goal_shell: float
    inner_shell: float
    dwell: float
    outer_exit: float
    inner_exit: float
    zone_exit: float
    near_goal_exit: float
    local_drift_penalty: float
    dwell_break: float
    ori_progress: float
    in_near_goal: float
    in_near_goal_shell: float
    in_inner_shell: float
    in_dwell: float
    zone_index: float
    dwell_count: float
    success_triggered_by_dwell: float
    success_latched: float
    reward_total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "action": self.action,
            "jerk": self.jerk,
            "adjust_penalty": self.adjust_penalty,
            "raw_action_penalty": self.raw_action_penalty,
            "reject_penalty": self.reject_penalty,
            "intervention": self.intervention,
            "clamp_or_projection": self.clamp_or_projection,
            "stall": self.stall,
            "ee_small_motion_penalty": self.ee_small_motion_penalty,
            "timeout_penalty": self.timeout_penalty,
            "reset_fail_penalty": self.reset_fail_penalty,
            "execution_fail_penalty": self.execution_fail_penalty,
            "timeout_or_reset": self.timeout_or_reset,
            "success_bonus": self.success_bonus,
            "near_goal": self.near_goal,
            "near_goal_shell": self.near_goal_shell,
            "inner_shell": self.inner_shell,
            "dwell": self.dwell,
            "outer_exit": self.outer_exit,
            "inner_exit": self.inner_exit,
            "zone_exit": self.zone_exit,
            "near_goal_exit": self.near_goal_exit,
            "local_drift_penalty": self.local_drift_penalty,
            "dwell_break": self.dwell_break,
            "ori_progress": self.ori_progress,
            "in_near_goal": self.in_near_goal,
            "in_near_goal_shell": self.in_near_goal_shell,
            "in_inner_shell": self.in_inner_shell,
            "in_dwell": self.in_dwell,
            "zone_index": self.zone_index,
            "dwell_count": self.dwell_count,
            "success_triggered_by_dwell": self.success_triggered_by_dwell,
            "success_latched": self.success_latched,
            "reward_total": self.reward_total,
        }


class RewardComposer:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._prev_in_near_goal: bool = False
        self._prev_in_near_goal_shell: bool = False
        self._prev_in_dwell: bool = False
        self._prev_zone_index: int = 0
        self._dwell_count: int = 0
        self._success_awarded: bool = False

    def reset_episode_state(self) -> None:
        self._prev_in_near_goal = False
        self._prev_in_near_goal_shell = False
        self._prev_in_dwell = False
        self._prev_zone_index = 0
        self._dwell_count = 0
        self._success_awarded = False

    def state_dict(self) -> dict[str, bool | int]:
        return {
            "prev_in_near_goal": bool(self._prev_in_near_goal),
            "prev_in_near_goal_shell": bool(self._prev_in_near_goal_shell),
            "prev_in_dwell": bool(self._prev_in_dwell),
            "prev_zone_index": int(self._prev_zone_index),
            "dwell_count": int(self._dwell_count),
            "success_awarded": bool(self._success_awarded),
        }

    @staticmethod
    def _norm(x: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(x, dtype=float)))

    @staticmethod
    def ee_error_norm(ee_pos_err: np.ndarray, ee_ori_err: np.ndarray) -> float:
        pos = float(np.linalg.norm(np.asarray(ee_pos_err, dtype=float)))
        ori = float(np.linalg.norm(np.asarray(ee_ori_err, dtype=float)))
        return pos + 0.5 * ori

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-60.0, min(60.0, float(x)))
        return 1.0 / (1.0 + math.exp(-x))

    def compute(
        self,
        prev_ee_pos_err: np.ndarray,
        prev_ee_ori_err: np.ndarray,
        curr_ee_pos_err: np.ndarray,
        curr_ee_ori_err: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        intervention: bool,
        clamp_or_projection: bool,
        done: bool,
        done_reason: str,
        q_before: np.ndarray | None = None,
        q_after: np.ndarray | None = None,
        effect_ratio: float | None = None,
        prev_in_near_goal: bool = False,
        dwell_count: int = 0,
        prev_success_latched: bool = False,
        action_raw: np.ndarray | None = None,
        action_exec: np.ndarray | None = None,
        rejected: bool = False,
    ) -> RewardTerms:
        del q_before, q_after, effect_ratio, prev_in_near_goal, dwell_count, prev_success_latched

        dpos_prev = self._norm(prev_ee_pos_err)
        dpos_curr = self._norm(curr_ee_pos_err)
        dori_prev = self._norm(prev_ee_ori_err)
        dori_curr = self._norm(curr_ee_ori_err)

        # 1) execution_fail override.
        if done and done_reason == "execution_fail":
            self._prev_in_near_goal = False
            self._prev_in_near_goal_shell = False
            self._prev_in_dwell = False
            self._prev_zone_index = 0
            self._dwell_count = 0
            self._success_awarded = False
            execution_fail_penalty = float(self.config.execution_fail_penalty)
            return RewardTerms(
                progress=0.0,
                action=0.0,
                jerk=0.0,
                adjust_penalty=0.0,
                raw_action_penalty=0.0,
                reject_penalty=0.0,
                intervention=0.0,
                clamp_or_projection=0.0,
                stall=0.0,
                ee_small_motion_penalty=0.0,
                timeout_penalty=0.0,
                reset_fail_penalty=0.0,
                execution_fail_penalty=execution_fail_penalty,
                timeout_or_reset=execution_fail_penalty,
                success_bonus=0.0,
                near_goal=0.0,
                near_goal_shell=0.0,
                inner_shell=0.0,
                dwell=0.0,
                outer_exit=0.0,
                inner_exit=0.0,
                zone_exit=0.0,
                near_goal_exit=0.0,
                local_drift_penalty=0.0,
                dwell_break=0.0,
                ori_progress=0.0,
                in_near_goal=0.0,
                in_near_goal_shell=0.0,
                in_inner_shell=0.0,
                in_dwell=0.0,
                zone_index=0.0,
                dwell_count=0.0,
                success_triggered_by_dwell=0.0,
                success_latched=0.0,
                reward_total=execution_fail_penalty,
            )

        # 2) position progress: asymmetric linear + log.
        delta_pos = dpos_prev - dpos_curr
        log_term = self.config.w_pos_progress_log * (
            math.log(dpos_prev + self.config.pos_log_eps)
            - math.log(dpos_curr + self.config.pos_log_eps)
        )
        if delta_pos >= 0.0:
            r_pos_progress = self.config.w_pos_progress_lin_toward * delta_pos + log_term
        else:
            close_distance = min(dpos_prev, dpos_curr)
            close_ratio = 0.0
            if self.config.outer_shell_pos_m > 0.0:
                close_ratio = max(0.0, 1.0 - (close_distance / self.config.outer_shell_pos_m))
            away_scale = 1.0 + (self.config.w_pos_progress_away_near_scale * close_ratio)
            r_pos_progress = (self.config.w_pos_progress_lin_away * away_scale * delta_pos) + log_term

        # 3) orientation progress (disabled in phase-1 by default).
        r_ori_progress = self.config.w_ori_progress * (dori_prev - dori_curr)

        # 4) smoothness (squared L2).
        action_exec_arr = np.asarray(action_exec if action_exec is not None else action, dtype=np.float32)
        action_raw_arr = np.asarray(action_raw if action_raw is not None else action, dtype=np.float32)
        a = action_exec_arr
        pa = np.asarray(prev_action, dtype=np.float32)
        r_action = self.config.w_action_norm * float(np.dot(a, a))
        da = a - pa
        r_jerk = self.config.w_jerk * float(np.dot(da, da))
        norm_denom = max(float(self.config.action_scale), 1e-8)
        action_adjust = (action_exec_arr - action_raw_arr) / norm_denom
        action_raw_norm = action_raw_arr / norm_denom
        r_adjust_penalty = -float(self.config.w_adjust) * float(np.dot(action_adjust, action_adjust))
        r_raw_action_penalty = -float(self.config.w_raw_action) * float(np.dot(action_raw_norm, action_raw_norm))
        r_reject_penalty = float(self.config.reject_penalty) if rejected else 0.0

        # 5) safety.
        r_intervention = self.config.w_intervention if intervention else 0.0
        r_clamp = self.config.w_clamp_projection if clamp_or_projection else 0.0

        # 6) near-goal / dwell.
        in_dwell_region = dpos_curr < self.config.dwell_pos_m
        in_inner_shell = (dpos_curr >= self.config.dwell_pos_m) and (dpos_curr < self.config.inner_shell_pos_m)
        in_outer_shell = (dpos_curr >= self.config.inner_shell_pos_m) and (dpos_curr < self.config.outer_shell_pos_m)
        in_near_goal = in_inner_shell or in_dwell_region
        in_near_goal_shell = in_outer_shell
        zone_index = 3 if in_dwell_region else (2 if in_inner_shell else (1 if in_outer_shell else 0))

        r_near_goal = self.config.near_goal_bonus if (in_inner_shell and not self._prev_in_near_goal) else 0.0

        if self.config.smooth_basin_enabled:
            temp = max(float(self.config.smooth_basin_temperature_m), 1e-6)
            outer_level = self._sigmoid((float(self.config.outer_shell_pos_m) - dpos_curr) / temp)
            inner_level = self._sigmoid((float(self.config.inner_shell_pos_m) - dpos_curr) / temp)
            # Phase-A dense basin: no hard reward cliff at shell boundaries.
            r_near_goal_shell = float(self.config.shell_bonus) * outer_level
            r_inner_shell = float(self.config.inner_shell_bonus) * inner_level
        else:
            outer_span = max(self.config.outer_shell_pos_m - self.config.inner_shell_pos_m, 1e-8)
            outer_closeness = 0.0
            if in_outer_shell:
                outer_closeness = max(
                    0.0,
                    min(1.0, (self.config.outer_shell_pos_m - dpos_curr) / outer_span),
                )
            r_near_goal_shell = self.config.shell_bonus * (1.0 + outer_closeness) if in_outer_shell else 0.0

            inner_span = max(self.config.inner_shell_pos_m - self.config.dwell_pos_m, 1e-8)
            inner_closeness = 0.0
            if in_inner_shell:
                inner_closeness = max(
                    0.0,
                    min(1.0, (self.config.inner_shell_pos_m - dpos_curr) / inner_span),
                )
            r_inner_shell = self.config.inner_shell_bonus * (1.0 + inner_closeness) if in_inner_shell else 0.0

        if in_dwell_region:
            self._dwell_count += 1
            r_dwell = self.config.dwell_bonus
        else:
            self._dwell_count = 0
            r_dwell = 0.0

        prev_zone_index = int(self._prev_zone_index)
        r_outer_exit = float(self.config.outer_exit_penalty) if (prev_zone_index == 1 and zone_index == 0) else 0.0
        r_inner_exit = float(self.config.inner_exit_penalty) if (prev_zone_index == 2 and zone_index < 2) else 0.0
        r_dwell_break = float(self.config.dwell_break_penalty) if (prev_zone_index == 3 and zone_index != 3) else 0.0
        r_zone_exit = r_outer_exit + r_inner_exit + r_dwell_break
        r_near_goal_exit = r_inner_exit
        drifting_within_basin = (prev_zone_index in {1, 2}) and (zone_index in {1, 2}) and (dpos_curr > dpos_prev)
        r_local_drift_penalty = (
            -float(self.config.drift_lambda) * float(dpos_curr - dpos_prev)
            if drifting_within_basin
            else 0.0
        )

        # 7) terminal.
        r_terminal = 0.0
        success_by_dwell = self._dwell_count >= self.config.success_dwell_steps
        just_succeeded = ((done and done_reason == "success") or success_by_dwell) and (not self._success_awarded)
        success_triggered_by_dwell = bool(success_by_dwell and just_succeeded)

        if just_succeeded:
            r_terminal += float(self.config.success_bonus)
            self._success_awarded = True
        elif done and done_reason == "timeout":
            r_terminal += float(self.config.timeout_penalty)
        elif done and done_reason == "reset_fail":
            r_terminal += float(self.config.reset_fail_penalty)

        r_success_bonus = float(self.config.success_bonus) if just_succeeded else 0.0
        r_timeout_penalty = float(self.config.timeout_penalty) if (done and done_reason == "timeout") else 0.0
        r_reset_fail_penalty = float(self.config.reset_fail_penalty) if (done and done_reason == "reset_fail") else 0.0
        r_execution_fail_penalty = 0.0
        self._prev_in_near_goal = in_near_goal
        self._prev_in_near_goal_shell = in_outer_shell
        self._prev_in_dwell = in_dwell_region
        self._prev_zone_index = int(zone_index)

        total = (
            r_pos_progress
            + r_ori_progress
            + r_action
            + r_jerk
            + r_adjust_penalty
            + r_raw_action_penalty
            + r_reject_penalty
            + r_intervention
            + r_clamp
            + r_near_goal
            + r_near_goal_shell
            + r_inner_shell
            + r_dwell
            + r_local_drift_penalty
            + r_zone_exit
            + r_terminal
        )

        return RewardTerms(
            progress=float(r_pos_progress),
            action=float(r_action),
            jerk=float(r_jerk),
            adjust_penalty=float(r_adjust_penalty),
            raw_action_penalty=float(r_raw_action_penalty),
            reject_penalty=float(r_reject_penalty),
            intervention=float(r_intervention),
            clamp_or_projection=float(r_clamp),
            stall=0.0,
            ee_small_motion_penalty=0.0,
            timeout_penalty=float(r_timeout_penalty),
            reset_fail_penalty=float(r_reset_fail_penalty),
            execution_fail_penalty=float(r_execution_fail_penalty),
            timeout_or_reset=float(r_terminal),
            success_bonus=float(r_success_bonus),
            near_goal=float(r_near_goal),
            near_goal_shell=float(r_near_goal_shell),
            inner_shell=float(r_inner_shell),
            dwell=float(r_dwell),
            outer_exit=float(r_outer_exit),
            inner_exit=float(r_inner_exit),
            zone_exit=float(r_zone_exit),
            near_goal_exit=float(r_near_goal_exit),
            local_drift_penalty=float(r_local_drift_penalty),
            dwell_break=float(r_dwell_break),
            ori_progress=float(r_ori_progress),
            in_near_goal=float(in_near_goal),
            in_near_goal_shell=float(in_outer_shell),
            in_inner_shell=float(in_inner_shell),
            in_dwell=float(in_dwell_region),
            zone_index=float(zone_index),
            dwell_count=float(self._dwell_count),
            success_triggered_by_dwell=float(success_triggered_by_dwell),
            success_latched=float(self._success_awarded),
            reward_total=float(total),
        )


class RewardTraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
