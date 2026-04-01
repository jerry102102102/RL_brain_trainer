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
    w_pos_progress_lin: float = 6.0
    w_pos_progress_log: float = 0.12
    pos_log_eps: float = 1e-3

    # Orientation progress disabled for phase-1.
    w_ori_progress: float = 0.0

    # Mild smoothness regularization (squared L2).
    w_action_norm: float = -0.002
    w_jerk: float = -0.001

    # Keep safety penalties but lighter.
    w_intervention: float = -0.10
    w_clamp_projection: float = -0.05

    # Terminal rewards/penalties.
    timeout_penalty: float = -0.2
    reset_fail_penalty: float = -1.5
    success_bonus: float = 3.0
    execution_fail_penalty: float = -2.0

    # Phase-1: disable these penalties.
    stall_penalty: float = 0.0
    ee_small_motion_penalty: float = 0.0

    # Near-goal / dwell shaping.
    near_goal_pos_m: float = 0.04
    near_goal_bonus: float = 0.05
    dwell_pos_m: float = 0.02
    dwell_ori_rad: float = 0.12
    dwell_bonus: float = 0.12
    dwell_steps_required: int = 5
    near_goal_exit_penalty: float = -0.10


@dataclass(frozen=True)
class RewardTerms:
    progress: float
    action: float
    jerk: float
    intervention: float
    clamp_or_projection: float
    stall: float
    ee_small_motion_penalty: float
    timeout_or_reset: float
    success_bonus: float
    near_goal: float
    dwell: float
    near_goal_exit: float
    ori_progress: float
    reward_total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "action": self.action,
            "jerk": self.jerk,
            "intervention": self.intervention,
            "clamp_or_projection": self.clamp_or_projection,
            "stall": self.stall,
            "ee_small_motion_penalty": self.ee_small_motion_penalty,
            "timeout_or_reset": self.timeout_or_reset,
            "success_bonus": self.success_bonus,
            "near_goal": self.near_goal,
            "dwell": self.dwell,
            "near_goal_exit": self.near_goal_exit,
            "ori_progress": self.ori_progress,
            "reward_total": self.reward_total,
        }


class RewardComposer:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        # Internal rollout state for near-goal/dwell shaping.
        self._prev_in_near_goal: bool = False
        self._dwell_count: int = 0

    @staticmethod
    def _norm(x: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(x, dtype=float)))

    def reset_episode_state(self) -> None:
        self._prev_in_near_goal = False
        self._dwell_count = 0

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
    ) -> RewardTerms:
        del q_before, q_after, effect_ratio  # unused in phase-1 ablation

        dpos_prev = self._norm(prev_ee_pos_err)
        dpos_curr = self._norm(curr_ee_pos_err)
        dori_prev = self._norm(prev_ee_ori_err)
        dori_curr = self._norm(curr_ee_ori_err)

        # 1) execution_fail override.
        if done and done_reason == "execution_fail":
            return RewardTerms(
                progress=0.0,
                action=0.0,
                jerk=0.0,
                intervention=0.0,
                clamp_or_projection=0.0,
                stall=0.0,
                ee_small_motion_penalty=0.0,
                timeout_or_reset=float(self.config.execution_fail_penalty),
                success_bonus=0.0,
                near_goal=0.0,
                dwell=0.0,
                near_goal_exit=0.0,
                ori_progress=0.0,
                reward_total=float(self.config.execution_fail_penalty),
            )

        # 2) position progress: linear + log.
        r_pos_progress = (
            self.config.w_pos_progress_lin * (dpos_prev - dpos_curr)
            + self.config.w_pos_progress_log
            * (
                math.log(dpos_prev + self.config.pos_log_eps)
                - math.log(dpos_curr + self.config.pos_log_eps)
            )
        )

        # 3) orientation progress (disabled in phase-1 by default).
        r_ori_progress = self.config.w_ori_progress * (dori_prev - dori_curr)

        # 4) smoothness (squared L2).
        a = np.asarray(action, dtype=np.float32)
        pa = np.asarray(prev_action, dtype=np.float32)
        r_action = self.config.w_action_norm * float(np.dot(a, a))
        da = a - pa
        r_jerk = self.config.w_jerk * float(np.dot(da, da))

        # 5) safety.
        r_intervention = self.config.w_intervention if intervention else 0.0
        r_clamp = self.config.w_clamp_projection if clamp_or_projection else 0.0

        # 6) near-goal / dwell.
        in_near_goal = dpos_curr < self.config.near_goal_pos_m
        in_dwell_region = (dpos_curr < self.config.dwell_pos_m) and (dori_curr < self.config.dwell_ori_rad)

        r_near_goal = self.config.near_goal_bonus if in_near_goal else 0.0

        if in_dwell_region:
            self._dwell_count += 1
            r_dwell = self.config.dwell_bonus
        else:
            self._dwell_count = 0
            r_dwell = 0.0

        r_near_goal_exit = self.config.near_goal_exit_penalty if (self._prev_in_near_goal and not in_near_goal) else 0.0

        # 7) terminal.
        r_terminal = 0.0
        success_by_dwell = self._dwell_count >= self.config.dwell_steps_required
        if done_reason == "success" or success_by_dwell:
            r_terminal += self.config.success_bonus
        elif done_reason == "timeout":
            r_terminal += self.config.timeout_penalty
        elif done_reason == "reset_fail":
            r_terminal += self.config.reset_fail_penalty

        total = (
            r_pos_progress
            + r_ori_progress
            + r_action
            + r_jerk
            + r_intervention
            + r_clamp
            + r_near_goal
            + r_dwell
            + r_near_goal_exit
            + r_terminal
        )

        self._prev_in_near_goal = in_near_goal

        return RewardTerms(
            progress=float(r_pos_progress),
            action=float(r_action),
            jerk=float(r_jerk),
            intervention=float(r_intervention),
            clamp_or_projection=float(r_clamp),
            stall=0.0,
            ee_small_motion_penalty=0.0,
            timeout_or_reset=float(r_terminal),
            success_bonus=float(self.config.success_bonus if (done_reason == "success" or success_by_dwell) else 0.0),
            near_goal=float(r_near_goal),
            dwell=float(r_dwell),
            near_goal_exit=float(r_near_goal_exit),
            ori_progress=float(r_ori_progress),
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
