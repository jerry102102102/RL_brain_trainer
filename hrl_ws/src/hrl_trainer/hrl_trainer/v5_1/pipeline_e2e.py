"""V5.1 end-to-end pipeline (real reward + minimal SAC)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .curriculum import CurriculumManager, resolve_stages
from .ee_fk import ee_pose6_from_q
from .gates import DEFAULT_GATE, GateEvaluator, write_gate_report
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig
from .pipeline_smoke import run_smoke
from .reward import RewardComposer, RewardConfig, RewardTraceWriter
from .runtime_ros2 import RuntimeROS2Adapter
from .training_report import write_training_report


RuntimeFactory = Callable[..., RuntimeROS2Adapter]
PolicyStepFn = Callable[[np.ndarray], tuple[np.ndarray, str] | tuple[np.ndarray, str, dict[str, Any]]]

_CONTROLLED_ACTION_DIM = 7
_OBS_DIM = 27
_NO_EFFECT_EPS = 1e-4
_NO_EFFECT_STREAK_LIMIT = 3
_HOME_Q = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
_PROGRESS_LOG_EVERY_STEPS = 5
_EXTERNAL_TASK_LIBRARY_RELATIVE = Path(
    "external/ENPM662_Group4_FinalProject/src/kitchen_robot_controller/kitchen_robot_controller/task_library.py"
)


@dataclass(frozen=True)
class TargetCurriculumStage:
    name: str
    pos_offset_min_m: float
    pos_offset_max_m: float
    ori_offset_min_deg: float
    ori_offset_max_deg: float


@dataclass
class TargetCurriculumState:
    stage_index: int = 0
    no_improvement_evals: int = 0
    best_eval_score: float = float("-inf")
    best_eval_episode: int = -1


class TargetCurriculumManager:
    def __init__(self, final_stage: TargetCurriculumStage, max_stage_index: int | None = None) -> None:
        self.stages = (
            TargetCurriculumStage("TC0", 0.08, 0.10, 0.0, 2.0),
            TargetCurriculumStage("TC1", 0.10, 0.13, 1.0, 4.0),
            final_stage,
        )
        if max_stage_index is None:
            self.max_stage_index = len(self.stages) - 1
        else:
            self.max_stage_index = max(0, min(int(max_stage_index), len(self.stages) - 1))
        self.state = TargetCurriculumState()
        self.history: list[dict[str, Any]] = []

    @property
    def current_stage(self) -> TargetCurriculumStage:
        return self.stages[self.state.stage_index]

    def record_eval(self, episode_index: int, eval_metrics: dict[str, Any], eval_score: float) -> dict[str, Any]:
        promoted = False
        previous_stage = self.current_stage.name

        if eval_score > self.state.best_eval_score + 1e-9:
            self.state.best_eval_score = float(eval_score)
            self.state.best_eval_episode = int(episode_index)
            self.state.no_improvement_evals = 0
        else:
            self.state.no_improvement_evals += 1

        can_promote = self.state.stage_index < self.max_stage_index
        if can_promote and self.state.stage_index == 0:
            shell_entry_rate = float(eval_metrics.get("true_basin_hit_rate", eval_metrics.get("shell_hit_rate", 0.0)))
            final_minus_min = float(eval_metrics.get("mean_final_minus_min", 1.0))
            if shell_entry_rate >= 0.70 and final_minus_min < 0.015:
                self.state.stage_index = 1
                promoted = True
        elif can_promote and self.state.stage_index == 1:
            dwell_hit_rate = float(eval_metrics.get("true_dwell_hit_rate", eval_metrics.get("dwell_hit_rate", 0.0)))
            success_rate = float(eval_metrics.get("success_rate", 0.0))
            if dwell_hit_rate >= 0.40 or success_rate > 0.0:
                self.state.stage_index = 2
                promoted = True

        event = {
            "episode_index": int(episode_index),
            "stage_before": previous_stage,
            "stage_after": self.current_stage.name,
            "promoted": bool(promoted),
            "eval_score": float(eval_score),
            "shell_hit_rate": float(eval_metrics.get("shell_hit_rate", 0.0)),
            "inner_shell_hit_rate": float(eval_metrics.get("inner_shell_hit_rate", 0.0)),
            "dwell_hit_rate": float(eval_metrics.get("dwell_hit_rate", 0.0)),
            "true_outer_hit_rate": float(eval_metrics.get("true_outer_hit_rate", 0.0)),
            "true_inner_hit_rate": float(eval_metrics.get("true_inner_hit_rate", 0.0)),
            "true_dwell_hit_rate": float(eval_metrics.get("true_dwell_hit_rate", 0.0)),
            "true_basin_hit_rate": float(eval_metrics.get("true_basin_hit_rate", 0.0)),
            "true_final_basin_rate": float(eval_metrics.get("true_final_basin_rate", 0.0)),
            "success_rate": float(eval_metrics.get("success_rate", 0.0)),
            "mean_final_minus_min": float(eval_metrics.get("mean_final_minus_min", 0.0)),
            "no_improvement_evals": int(self.state.no_improvement_evals),
            "max_stage_index": int(self.max_stage_index),
            "promotion_locked": bool(not can_promote),
        }
        self.history.append(event)
        return event

    def to_artifact(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "current_stage": asdict(self.current_stage),
            "stages": [asdict(s) for s in self.stages],
            "max_stage_index": int(self.max_stage_index),
            "history": list(self.history),
        }


@dataclass(frozen=True)
class EntropyAnnealStage:
    name: str
    ratio: float
    target_entropy: float


@dataclass
class EntropyAnnealState:
    mode: str = "off"
    stage_index: int = 0
    baseline_target_entropy: float = 0.0
    current_target_entropy: float = 0.0
    baseline_det_action_l2: float | None = None
    baseline_det_raw_norm: float | None = None
    baseline_action_l2_ratio: float | None = None
    baseline_raw_norm_ratio: float | None = None


class EntropyAnnealManager:
    def __init__(
        self,
        *,
        mode: str,
        baseline_target_entropy: float,
        ratios: list[float],
        stage_names: list[str],
        fixed_episode_thresholds: list[int],
        min_episode: int,
        window: int,
        max_stage_index: int | None = None,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in {"off", "event", "fixed"}:
            raise ValueError("entropy_anneal_mode must be one of: off|event|fixed")
        ratios = [float(r) for r in ratios if float(r) > 0.0]
        if not ratios:
            ratios = [1.0]
        if abs(float(ratios[0]) - 1.0) > 1e-9:
            ratios = [1.0, *ratios]
        names = list(stage_names)
        while len(names) < len(ratios):
            names.append(chr(ord("A") + len(names)))
        self.stages = [
            EntropyAnnealStage(
                name=str(names[idx]),
                ratio=float(ratio),
                target_entropy=float(baseline_target_entropy) * float(ratio),
            )
            for idx, ratio in enumerate(ratios)
        ]
        self.state = EntropyAnnealState(
            mode=normalized_mode,
            stage_index=0,
            baseline_target_entropy=float(baseline_target_entropy),
            current_target_entropy=float(self.stages[0].target_entropy),
        )
        self.fixed_episode_thresholds = [max(1, int(v)) for v in fixed_episode_thresholds]
        self.min_episode = max(1, int(min_episode))
        self.window = max(1, int(window))
        if max_stage_index is None:
            self.max_stage_index = len(self.stages) - 1
        else:
            self.max_stage_index = max(0, min(int(max_stage_index), len(self.stages) - 1))
        self.history: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.state.mode != "off" and self.max_stage_index > 0 and len(self.stages) > 1

    @property
    def current_stage(self) -> EntropyAnnealStage:
        return self.stages[self.state.stage_index]

    def apply_to_agent(self, agent: Any) -> None:
        if hasattr(agent, "set_target_entropy"):
            agent.set_target_entropy(float(self.current_stage.target_entropy))
        else:
            agent.target_entropy = float(self.current_stage.target_entropy)
        self.state.current_target_entropy = float(self.current_stage.target_entropy)

    def observe_eval(self, eval_metrics: dict[str, Any]) -> None:
        det_action = float(eval_metrics.get("det_action_l2_mean", eval_metrics.get("final_action_l2_mean", 0.0)))
        det_raw = float(eval_metrics.get("det_raw_norm_mean", eval_metrics.get("raw_norm_mean", 0.0)))
        action_ratio = float(eval_metrics.get("det_action_l2_over_stoch_action_l2", 0.0))
        raw_ratio = float(eval_metrics.get("det_raw_norm_over_stoch_raw_norm", 0.0))
        if self.state.baseline_det_action_l2 is None:
            self.state.baseline_det_action_l2 = det_action
        if self.state.baseline_det_raw_norm is None:
            self.state.baseline_det_raw_norm = det_raw
        if self.state.baseline_action_l2_ratio is None and action_ratio > 0.0:
            self.state.baseline_action_l2_ratio = action_ratio
        if self.state.baseline_raw_norm_ratio is None and raw_ratio > 0.0:
            self.state.baseline_raw_norm_ratio = raw_ratio

    def maybe_advance(
        self,
        *,
        episode_index: int,
        agent: Any,
        recent_train_metrics: dict[str, float],
        eval_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.enabled or self.state.stage_index >= min(self.max_stage_index, len(self.stages) - 1):
            return None
        self.observe_eval(eval_metrics)
        completed_episode = int(episode_index) + 1
        should_advance = False
        reason = ""
        if self.state.mode == "fixed":
            threshold = (
                self.fixed_episode_thresholds[self.state.stage_index]
                if self.state.stage_index < len(self.fixed_episode_thresholds)
                else None
            )
            if threshold is not None and completed_episode >= int(threshold):
                should_advance = True
                reason = f"fixed_episode>={int(threshold)}"
        elif self.state.stage_index == 0:
            train_basin = float(recent_train_metrics.get("true_basin_hit_rate", 0.0))
            train_inner = float(recent_train_metrics.get("true_inner_hit_rate", 0.0))
            train_reject = float(recent_train_metrics.get("reject_rate", 0.0))
            if (
                completed_episode >= self.min_episode
                and train_basin >= 0.35
                and train_inner >= 0.08
                and train_reject <= 0.05
            ):
                should_advance = True
                reason = (
                    f"event_train_basin={train_basin:.3f}_inner={train_inner:.3f}_"
                    f"reject={train_reject:.3f}"
                )
        else:
            det_outer = float(eval_metrics.get("det_true_outer_hit_rate", eval_metrics.get("true_outer_hit_rate", 0.0)))
            det_basin = float(eval_metrics.get("det_true_basin_hit_rate", eval_metrics.get("true_basin_hit_rate", 0.0)))
            det_action = float(eval_metrics.get("det_action_l2_mean", eval_metrics.get("final_action_l2_mean", 0.0)))
            det_raw = float(eval_metrics.get("det_raw_norm_mean", eval_metrics.get("raw_norm_mean", 0.0)))
            action_ratio = float(eval_metrics.get("det_action_l2_over_stoch_action_l2", 0.0))
            raw_ratio = float(eval_metrics.get("det_raw_norm_over_stoch_raw_norm", 0.0))
            baseline_action = max(float(self.state.baseline_det_action_l2 or 0.0), 1e-8)
            baseline_raw = max(float(self.state.baseline_det_raw_norm or 0.0), 1e-8)
            baseline_action_ratio = max(float(self.state.baseline_action_l2_ratio or 0.0), 1e-8)
            baseline_raw_ratio = max(float(self.state.baseline_raw_norm_ratio or 0.0), 1e-8)
            if det_outer > 0.0 or det_basin > 0.0:
                should_advance = True
                reason = f"event_det_outer_or_basin={det_outer:.3f}/{det_basin:.3f}"
            elif det_action / baseline_action >= 2.0:
                should_advance = True
                reason = f"event_det_action_l2_x{det_action / baseline_action:.2f}"
            elif det_raw / baseline_raw >= 2.0:
                should_advance = True
                reason = f"event_det_raw_norm_x{det_raw / baseline_raw:.2f}"
            elif action_ratio / baseline_action_ratio >= 2.0:
                should_advance = True
                reason = f"event_action_ratio_x{action_ratio / baseline_action_ratio:.2f}"
            elif raw_ratio / baseline_raw_ratio >= 2.0:
                should_advance = True
                reason = f"event_raw_ratio_x{raw_ratio / baseline_raw_ratio:.2f}"

        if not should_advance:
            return None

        previous_stage = self.current_stage
        self.state.stage_index += 1
        self.apply_to_agent(agent)
        event = {
            "episode": int(episode_index),
            "episode_completed": int(completed_episode),
            "stage_before": previous_stage.name,
            "stage_after": self.current_stage.name,
            "target_entropy_before": float(previous_stage.target_entropy),
            "target_entropy_after": float(self.current_stage.target_entropy),
            "ratio_before": float(previous_stage.ratio),
            "ratio_after": float(self.current_stage.ratio),
            "reason": str(reason),
            "mode": str(self.state.mode),
            "alpha": float(getattr(agent, "alpha", 0.0)),
            "recent_train_metrics": dict(recent_train_metrics),
            "eval_metrics": dict(eval_metrics),
        }
        self.history.append(event)
        return event

    def to_artifact(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "current_stage": asdict(self.current_stage),
            "stages": [asdict(stage) for stage in self.stages],
            "fixed_episode_thresholds": list(self.fixed_episode_thresholds),
            "min_episode": int(self.min_episode),
            "window": int(self.window),
            "max_stage_index": int(self.max_stage_index),
            "history": list(self.history),
            "enabled": bool(self.enabled),
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_ee_target_from_external_task(
    *,
    prop: str = "tray",
    src_idx: int = 2,
    dst_idx: int = 7,
    waypoint_index: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    external_file = _project_root() / _EXTERNAL_TASK_LIBRARY_RELATIVE
    spec = importlib.util.spec_from_file_location("v5_1_external_task_library", external_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load external task library spec: {external_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    task_lib = module.MoveTaskLibrary()
    trajectory = task_lib.move_from_to(prop=prop, src_idx=int(src_idx), dst_idx=int(dst_idx))
    if not trajectory:
        raise RuntimeError("external task trajectory is empty")

    idx = max(0, min(int(waypoint_index), len(trajectory) - 1))
    target_T = np.asarray(trajectory[idx], dtype=float)
    R = target_T[:3, :3]
    ee_target = np.array(
        [
            target_T[0, 3],
            target_T[1, 3],
            target_T[2, 3],
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)),
            np.arctan2(R[1, 0], R[0, 0]),
        ],
        dtype=float,
    )
    source = {
        "provider": "external_task_library.MoveTaskLibrary.move_from_to",
        "external_file": str(external_file),
        "prop": str(prop),
        "src_idx": int(src_idx),
        "dst_idx": int(dst_idx),
        "waypoint_index": int(idx),
        "trajectory_len": int(len(trajectory)),
    }
    return ee_target, source


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = np.asarray(rng.normal(size=3), dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def _resolve_near_home_ee_target(
    *,
    home_q: np.ndarray,
    profile: str = "s0_bootstrap",
    pos_offset_min_m: float = 0.22,
    pos_offset_max_m: float = 0.30,
    ori_offset_min_deg: float = 5.0,
    ori_offset_max_deg: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    home_q = np.asarray(home_q, dtype=float)
    home_ee = _ee_pose_from_q(home_q)
    rng = rng or np.random.default_rng()

    pos_mag = float(rng.uniform(float(pos_offset_min_m), float(pos_offset_max_m)))
    ori_mag_deg = float(rng.uniform(float(ori_offset_min_deg), float(ori_offset_max_deg)))
    ori_mag = float(np.deg2rad(ori_mag_deg))

    pos_dir = _sample_unit_vector(rng)
    if pos_dir[2] > 0.0:
        pos_dir[2] = -pos_dir[2]
    ori_dir = _sample_unit_vector(rng)

    delta_pos = pos_dir * pos_mag
    delta_ori = ori_dir * ori_mag

    ee_target = home_ee.copy()
    ee_target[:3] = ee_target[:3] + delta_pos
    ee_target[3:6] = _wrap_to_pi(ee_target[3:6] + delta_ori)

    source = {
        "provider": "near_home_randomized",
        "profile": str(profile),
        "home_q": home_q.tolist(),
        "home_ee": home_ee.tolist(),
        "target_delta_pos": delta_pos.tolist(),
        "target_delta_ori": delta_ori.tolist(),
        "target_delta_pos_l2": float(np.linalg.norm(delta_pos)),
        "target_delta_ori_l2": float(np.linalg.norm(delta_ori)),
        "z_not_above_home": True,
        "pos_offset_min_m": float(pos_offset_min_m),
        "pos_offset_max_m": float(pos_offset_max_m),
        "ori_offset_min_deg": float(ori_offset_min_deg),
        "ori_offset_max_deg": float(ori_offset_max_deg),
    }
    return ee_target, source


def _build_fixed_eval_suite(
    *,
    suite_size: int,
    suite_seed: int,
    target_mode: str,
    action_stage_name: str,
    target_curriculum_stage_name: str,
    near_home_profile: str,
    near_home_pos_offset_min_m: float,
    near_home_pos_offset_max_m: float,
    near_home_ori_offset_min_deg: float,
    near_home_ori_offset_max_deg: float,
    external_ee_target: np.ndarray,
    external_ee_target_source: dict[str, Any],
) -> dict[str, Any]:
    resolved_target_mode = str(target_mode)
    if resolved_target_mode == "auto":
        resolved_target_mode = "near_home" if str(action_stage_name) == "S0_B" else "external"

    rng = np.random.default_rng(int(suite_seed))
    targets: list[dict[str, Any]] = []
    fixed_home_q = np.asarray(_HOME_Q, dtype=float)
    for idx in range(max(1, int(suite_size))):
        if resolved_target_mode == "near_home":
            ee_target, ee_target_source = _resolve_near_home_ee_target(
                home_q=fixed_home_q,
                profile=near_home_profile,
                pos_offset_min_m=near_home_pos_offset_min_m,
                pos_offset_max_m=near_home_pos_offset_max_m,
                ori_offset_min_deg=near_home_ori_offset_min_deg,
                ori_offset_max_deg=near_home_ori_offset_max_deg,
                rng=rng,
            )
        else:
            ee_target = np.asarray(external_ee_target, dtype=float).copy()
            ee_target_source = dict(external_ee_target_source)
        targets.append(
            {
                "episode_index": int(idx),
                "ee_target": np.asarray(ee_target, dtype=float).tolist(),
                "ee_target_source": ee_target_source,
            }
        )

    return {
        "suite_id": (
            f"det_eval_{resolved_target_mode}_{int(suite_seed)}_"
            f"{str(target_curriculum_stage_name)}_{max(1, int(suite_size))}"
        ),
        "suite_seed": int(suite_seed),
        "suite_size": max(1, int(suite_size)),
        "resolved_target_mode": resolved_target_mode,
        "action_stage_name": str(action_stage_name),
        "target_curriculum_stage_name": str(target_curriculum_stage_name),
        "near_home_profile": str(near_home_profile),
        "near_home_pos_offset_min_m": float(near_home_pos_offset_min_m),
        "near_home_pos_offset_max_m": float(near_home_pos_offset_max_m),
        "near_home_ori_offset_min_deg": float(near_home_ori_offset_min_deg),
        "near_home_ori_offset_max_deg": float(near_home_ori_offset_max_deg),
        "targets": targets,
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _eval_score(metrics: dict[str, Any]) -> float:
    return (
        2.0 * float(metrics.get("success_rate", 0.0))
        + 1.0 * float(metrics.get("dwell_hit_rate", 0.0))
        - float(metrics.get("mean_final_dpos", 0.0))
        - float(metrics.get("mean_final_minus_min", 0.0))
    )


def _checkpoint_score(metrics: dict[str, Any]) -> float:
    return (
        3.0 * float(metrics.get("det_success_rate", 0.0))
        - 1.0 * float(metrics.get("mean_final_dpos", 0.0))
        - 1.0 * float(metrics.get("regression_rate", 0.0))
        - 1.0 * float(metrics.get("mean_final_minus_min", 0.0))
    )


def _reward_config_for_profile(profile: str, *, action_scale: float) -> RewardConfig:
    base = RewardConfig(action_scale=float(action_scale))
    normalized = str(profile or "default").strip().lower()
    if normalized in {"default", "hprs"}:
        return base
    if normalized in {"phase_a", "phase_a_bootstrap", "bootstrap"}:
        return replace(
            base,
            # Phase A: stronger dense reachability signal, less early over-constraint.
            w_pos_progress_lin_toward=7.0,
            w_pos_progress_lin_away=6.0,
            w_pos_progress_away_near_scale=1.5,
            smooth_basin_enabled=True,
            smooth_basin_temperature_m=0.015,
            shell_bonus=0.06,
            near_goal_shell_bonus=0.06,
            inner_shell_bonus=0.08,
            dwell_bonus=0.18,
            success_dwell_steps=2,
            dwell_steps_required=2,
            outer_exit_penalty=-0.05,
            inner_exit_penalty=-0.10,
            near_goal_exit_penalty=-0.10,
            dwell_break_penalty=-0.15,
            drift_lambda=3.0,
            timeout_penalty=-0.10,
        )
    if normalized in {"phase_a_v2", "phase_a_bootstrap_v2", "bootstrap_v2"}:
        return replace(
            base,
            # Phase A v2: keep a smooth outer basin, but make the center clearly more valuable.
            w_pos_progress_lin_toward=7.0,
            w_pos_progress_lin_away=8.0,
            w_pos_progress_away_near_scale=2.0,
            smooth_basin_enabled=True,
            smooth_basin_temperature_m=0.012,
            shell_bonus=0.04,
            near_goal_shell_bonus=0.04,
            inner_shell_bonus=0.14,
            dwell_bonus=0.25,
            success_dwell_steps=2,
            dwell_steps_required=2,
            outer_exit_penalty=-0.08,
            inner_exit_penalty=-0.16,
            near_goal_exit_penalty=-0.16,
            dwell_break_penalty=-0.24,
            drift_lambda=6.0,
            timeout_penalty=-0.15,
        )
    raise ValueError("reward_profile must be one of: default|phase_a_bootstrap|phase_a_bootstrap_v2")


def _schedule_exploration_scale(
    current_scale: float,
    *,
    total_successes: int,
    best_min_dpos: float,
    det_success_rate: float,
) -> tuple[float, str | None]:
    scale = float(current_scale)
    if scale > 0.45 + 1e-9:
        if int(total_successes) >= 5:
            return 0.45, "train_success>=5"
        if float(det_success_rate) >= 0.10:
            return 0.45, "det_success_rate>=0.10"
        if float(best_min_dpos) <= 0.020:
            return 0.45, "best_min_dpos<=0.020"
    return scale, None


def _maybe_schedule_exploration_scale(
    disable_exploration_schedule: bool,
    current_scale: float,
    *,
    total_successes: int,
    best_min_dpos: float,
    det_success_rate: float,
) -> tuple[float, str | None]:
    if bool(disable_exploration_schedule):
        return float(current_scale), None
    return _schedule_exploration_scale(
        current_scale,
        total_successes=total_successes,
        best_min_dpos=best_min_dpos,
        det_success_rate=det_success_rate,
    )


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _progress_logging_enabled() -> bool:
    quiet = os.environ.get("HRL_TRAINER_QUIET_PROGRESS", "").strip().lower()
    if quiet in {"1", "true", "yes", "on"}:
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _progress_log(message: str) -> None:
    if not _progress_logging_enabled():
        return
    print(f"[v5.1][{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _ee_pose_from_q(q: np.ndarray) -> np.ndarray:
    return ee_pose6_from_q(np.asarray(q, dtype=float))


def _ee_errors(ee_pose: np.ndarray, ee_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ee_pose = np.asarray(ee_pose, dtype=float)
    ee_target = np.asarray(ee_target, dtype=float)
    return ee_target[:3] - ee_pose[:3], ee_target[3:6] - ee_pose[3:6]


def _obs_from_state(q: np.ndarray, dq: np.ndarray, ee_pose_err: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    return np.concatenate([q, dq, ee_pose_err, prev_action], axis=0)


def _scaled_action_norm(action: np.ndarray, action_scale: float) -> float:
    denom = max(float(action_scale), 1e-8)
    arr = np.asarray(action, dtype=float) / denom
    return float(np.linalg.norm(arr))


def _dpos_zone(dpos: float, reward_config: RewardConfig) -> str:
    d = float(dpos)
    if d < float(reward_config.dwell_pos_m):
        return "dwell"
    if d < float(reward_config.inner_shell_pos_m):
        return "inner"
    if d < float(reward_config.outer_shell_pos_m):
        return "outer"
    return "outside"


def _dpos_zone_metrics(min_dpos: float, final_dpos: float, reward_config: RewardConfig) -> dict[str, Any]:
    min_zone = _dpos_zone(min_dpos, reward_config)
    final_zone = _dpos_zone(final_dpos, reward_config)
    return {
        "true_min_zone": min_zone,
        "true_final_zone": final_zone,
        "true_outer_hit": bool(min_zone == "outer"),
        "true_inner_hit": bool(min_zone == "inner"),
        "true_dwell_hit": bool(min_zone == "dwell"),
        "true_basin_hit": bool(min_zone in {"outer", "inner", "dwell"}),
        "true_inner_or_dwell_hit": bool(min_zone in {"inner", "dwell"}),
        "true_final_outer": bool(final_zone == "outer"),
        "true_final_inner": bool(final_zone == "inner"),
        "true_final_dwell": bool(final_zone == "dwell"),
        "true_final_basin": bool(final_zone in {"outer", "inner", "dwell"}),
        "true_final_inner_or_dwell": bool(final_zone in {"inner", "dwell"}),
    }


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _checkpoint_layout(artifact_root: Path) -> dict[str, Path]:
    train_root = artifact_root / "train"
    return {
        "latest": train_root / "checkpoint_latest.pt",
        "final": train_root / "checkpoint_final.pt",
        "best": train_root / "checkpoint_best.pt",
    }


def _checkpoint_candidates(artifact_root: Path) -> list[Path]:
    layout = _checkpoint_layout(artifact_root)
    return [layout["best"], layout["latest"], layout["final"]]


def _load_agent_checkpoint(agent: Any, checkpoint_path: Path) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.q1.load_state_dict(payload["q1_state_dict"])
    agent.q2.load_state_dict(payload["q2_state_dict"])
    agent.q1_target.load_state_dict(payload["q1_target_state_dict"])
    agent.q2_target.load_state_dict(payload["q2_target_state_dict"])
    agent.actor_optim.load_state_dict(payload["actor_optim_state_dict"])
    agent.q1_optim.load_state_dict(payload["q1_optim_state_dict"])
    agent.q2_optim.load_state_dict(payload["q2_optim_state_dict"])
    agent.log_alpha = torch.tensor(
        float(payload.get("log_alpha", np.log(max(float(payload.get("alpha", 0.2)), 1e-8)))),
        dtype=torch.float32,
        device=agent.device,
        requires_grad=True,
    )
    agent.alpha_optim = torch.optim.Adam([agent.log_alpha], lr=agent.cfg.lr_alpha)
    if "alpha_optim_state_dict" in payload:
        agent.alpha_optim.load_state_dict(payload["alpha_optim_state_dict"])
    if "target_entropy" in payload:
        if hasattr(agent, "set_target_entropy"):
            agent.set_target_entropy(float(payload["target_entropy"]))
        else:
            agent.target_entropy = float(payload["target_entropy"])

    agent.env_steps_collected = int(payload.get("env_steps_collected", 0))
    agent.updates_applied = int(payload.get("updates_applied", 0))
    agent.batch_draw_count = int(payload.get("batch_draw_count", 0))
    agent.actor_update_count = int(payload.get("actor_update_count", 0))
    agent.critic_update_count = int(payload.get("critic_update_count", 0))
    agent.alpha_update_count = int(payload.get("alpha_update_count", 0))
    agent.distill_update_count = int(payload.get("distill_update_count", 0))
    agent.distill_skip_count = int(payload.get("distill_skip_count", 0))
    agent.last_actor_hash = payload.get("last_actor_hash")
    agent.last_critic_hash = payload.get("last_critic_hash")


def _save_agent_checkpoint(
    agent: Any,
    checkpoint_path: Path,
    run_id: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp_ns": time.time_ns(),
        "actor_state_dict": agent.actor.state_dict(),
        "q1_state_dict": agent.q1.state_dict(),
        "q2_state_dict": agent.q2.state_dict(),
        "q1_target_state_dict": agent.q1_target.state_dict(),
        "q2_target_state_dict": agent.q2_target.state_dict(),
        "actor_optim_state_dict": agent.actor_optim.state_dict(),
        "q1_optim_state_dict": agent.q1_optim.state_dict(),
        "q2_optim_state_dict": agent.q2_optim.state_dict(),
        "alpha_optim_state_dict": agent.alpha_optim.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu().item()),
        "alpha": float(agent.alpha),
        "target_entropy": float(getattr(agent, "target_entropy", 0.0)),
        "env_steps_collected": int(agent.env_steps_collected),
        "updates_applied": int(agent.updates_applied),
        "batch_draw_count": int(agent.batch_draw_count),
        "actor_update_count": int(agent.actor_update_count),
        "critic_update_count": int(agent.critic_update_count),
        "alpha_update_count": int(agent.alpha_update_count),
        "distill_update_count": int(getattr(agent, "distill_update_count", 0)),
        "distill_skip_count": int(getattr(agent, "distill_skip_count", 0)),
        "last_actor_hash": agent.last_actor_hash,
        "last_critic_hash": agent.last_critic_hash,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def _read_agent_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return dict(payload.get("metadata", {}) or {})


def _parse_gap_eval_scales(spec: str | None) -> list[dict[str, Any]]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"det", "deterministic", "mean"}:
            label = "deterministic"
            scale = 0.0
            stochastic = False
        else:
            scale = float(token)
            if scale <= 0.0:
                label = "deterministic"
                stochastic = False
                scale = 0.0
            else:
                label = f"noise{int(round(scale * 100.0)):03d}"
                stochastic = True
        if label in seen:
            continue
        seen.add(label)
        out.append(
            {
                "label": label,
                "stochastic": bool(stochastic),
                "exploration_std_scale": float(scale),
            }
        )
    return out


def _parse_float_list(spec: str | None, default: list[float]) -> list[float]:
    raw = str(spec or "").strip()
    if not raw:
        return list(default)
    out: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        out.append(float(token))
    return out or list(default)


def _parse_int_list(spec: str | None, default: list[int]) -> list[int]:
    raw = str(spec or "").strip()
    if not raw:
        return list(default)
    out: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        out.append(int(token))
    return out or list(default)


def _parse_stage_names(spec: str | None, count: int) -> list[str]:
    raw = str(spec or "").strip()
    names = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    while len(names) < max(1, int(count)):
        names.append(chr(ord("A") + len(names)))
    return names[: max(1, int(count))]


def _recent_episode_metrics(episode_outputs: list[dict[str, Any]], window: int) -> dict[str, float]:
    recent = episode_outputs[-max(1, int(window)) :]
    if not recent:
        return {
            "episodes": 0.0,
            "success_rate": 0.0,
            "true_basin_hit_rate": 0.0,
            "true_outer_hit_rate": 0.0,
            "true_inner_hit_rate": 0.0,
            "true_dwell_hit_rate": 0.0,
            "reject_rate": 0.0,
            "execution_fail_rate": 0.0,
            "mean_final_dpos": 0.0,
            "mean_final_minus_min": 0.0,
        }
    return {
        "episodes": float(len(recent)),
        "success_rate": float(np.mean([float(ep.get("success_rate", 0.0)) for ep in recent])),
        "true_basin_hit_rate": float(np.mean([1.0 if ep.get("true_basin_hit", False) else 0.0 for ep in recent])),
        "true_outer_hit_rate": float(np.mean([1.0 if ep.get("true_outer_hit", False) else 0.0 for ep in recent])),
        "true_inner_hit_rate": float(np.mean([1.0 if ep.get("true_inner_hit", False) else 0.0 for ep in recent])),
        "true_dwell_hit_rate": float(np.mean([1.0 if ep.get("true_dwell_hit", False) else 0.0 for ep in recent])),
        "reject_rate": float(np.mean([float(ep.get("reject_rate", 0.0)) for ep in recent])),
        "execution_fail_rate": float(np.mean([1.0 if ep.get("done_reason") == "execution_fail" else 0.0 for ep in recent])),
        "mean_final_dpos": float(np.mean([float(ep.get("final_dpos", 0.0)) for ep in recent])),
        "mean_final_minus_min": float(np.mean([float(ep.get("final_dpos_minus_min_dpos", 0.0)) for ep in recent])),
    }


def _recent_train_update_metrics(train_metrics: list[dict[str, Any]], window: int) -> dict[str, float]:
    recent = train_metrics[-max(1, int(window)) :]
    if not recent:
        return {
            "updates": 0.0,
            "distill_loss": 0.0,
            "distill_good_count": 0.0,
            "distill_good_fraction": 0.0,
            "distill_quality_mean": 0.0,
            "distill_advantage_mean": 0.0,
            "distill_mean_action_l2": 0.0,
            "distill_target_action_l2": 0.0,
            "active_distill_lambda": 0.0,
        }
    return {
        "updates": float(len(recent)),
        "distill_loss": float(np.mean([float(m.get("distill_loss", 0.0)) for m in recent])),
        "distill_good_count": float(np.mean([float(m.get("distill_good_count", 0.0)) for m in recent])),
        "distill_good_fraction": float(np.mean([float(m.get("distill_good_fraction", 0.0)) for m in recent])),
        "distill_quality_mean": float(np.mean([float(m.get("distill_quality_mean", 0.0)) for m in recent])),
        "distill_advantage_mean": float(np.mean([float(m.get("distill_advantage_mean", 0.0)) for m in recent])),
        "distill_mean_action_l2": float(np.mean([float(m.get("distill_mean_action_l2", 0.0)) for m in recent])),
        "distill_target_action_l2": float(np.mean([float(m.get("distill_target_action_l2", 0.0)) for m in recent])),
        "active_distill_lambda": float(np.mean([float(m.get("distill_active_lambda", 0.0)) for m in recent])),
    }


def _vector_l2(value: Any) -> float | None:
    if not isinstance(value, list):
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(np.linalg.norm(arr))


def _reward_trace_action_stats(path: Path) -> dict[str, float]:
    if not Path(path).exists():
        return {}
    accum: dict[str, list[float]] = {
        "final_action_l2": [],
        "mu_l2": [],
        "std_l2": [],
        "std_scaled_l2": [],
        "noise_l2": [],
        "pre_tanh_l2": [],
        "post_tanh_l2": [],
        "raw_norm": [],
        "exec_norm": [],
        "delta_norm": [],
        "pre_tanh_abs_max": [],
        "post_tanh_abs_max": [],
        "saturated_dims": [],
    }
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            debug = row.get("policy_debug") or {}
            for key, debug_key in (
                ("final_action_l2", "final_action"),
                ("mu_l2", "mu"),
                ("std_l2", "std"),
                ("std_scaled_l2", "std_scaled"),
                ("noise_l2", "noise"),
                ("pre_tanh_l2", "pre_tanh"),
                ("post_tanh_l2", "post_tanh"),
            ):
                value = _vector_l2(debug.get(debug_key))
                if value is not None:
                    accum[key].append(value)
            for key in ("raw_norm", "exec_norm", "delta_norm"):
                if row.get(key) is not None:
                    accum[key].append(float(row.get(key)))
            for key in ("pre_tanh_abs_max", "post_tanh_abs_max", "saturated_dims"):
                if debug.get(key) is not None:
                    accum[key].append(float(debug.get(key)))
    return {
        f"{key}_mean": float(np.mean(values))
        for key, values in accum.items()
        if values
    } | {
        f"{key}_max": float(np.max(values))
        for key, values in accum.items()
        if values and key in {"final_action_l2", "raw_norm", "exec_norm", "delta_norm", "pre_tanh_abs_max"}
    }


def _controlled_joint_indices(runtime_joint_names: list[str]) -> list[int]:
    indices = list(range(len(runtime_joint_names)))
    if len(indices) != _CONTROLLED_ACTION_DIM:
        raise ValueError(
            "runtime_joint_names must resolve to exactly 7 controllable joints "
            f"(including Rack_joint); got {len(indices)} from {runtime_joint_names}"
        )
    return indices


def _curriculum_stage_by_name(curriculum: Any, stage_name: str) -> Any:
    for stage in getattr(curriculum, "stages", []):
        if getattr(stage, "name", None) == str(stage_name):
            return stage
    return curriculum.current_stage


def _expand_cmd_q(q_before_full: np.ndarray, controlled_indices: list[int], q_des_controlled: np.ndarray) -> np.ndarray:
    cmd_full = np.asarray(q_before_full, dtype=float).copy()
    cmd_full[np.asarray(controlled_indices, dtype=int)] = np.asarray(q_des_controlled, dtype=float)
    return cmd_full


def _reset_episode_home(
    runtime: RuntimeROS2Adapter,
    controlled_indices: list[int],
    home_q: np.ndarray,
    reset_near_home_eps: float,
) -> dict[str, Any]:
    q_before_full = runtime.read_q()
    controlled_idx_np = np.asarray(controlled_indices, dtype=int)
    q_before_controlled = np.asarray(q_before_full, dtype=float)[controlled_idx_np]
    home_q = np.asarray(home_q, dtype=float)
    near_home_l2 = float(np.linalg.norm(q_before_controlled - home_q))
    reset_skipped_near_home = bool(near_home_l2 < float(reset_near_home_eps))

    if reset_skipped_near_home:
        return {
            "accepted": True,
            "result_status": "success",
            "execution_ok": True,
            "fail_reason": "none",
            "command_path": "skipped_near_home",
            "home_q": home_q.tolist(),
            "q_after": np.asarray(q_before_full, dtype=float).tolist(),
            "q_before": np.asarray(q_before_full, dtype=float).tolist(),
            "reset_skipped_near_home": True,
            "reset_near_home_l2": near_home_l2,
            "runtime": None,
        }

    cmd_q_full = _expand_cmd_q(q_before_full=q_before_full, controlled_indices=controlled_indices, q_des_controlled=home_q)
    out = runtime.step(cmd_q_full)
    return {
        "accepted": bool(out.get("accepted", False)),
        "result_status": str(out.get("result_status", "fail")),
        "execution_ok": bool(out.get("execution_ok", False)),
        "fail_reason": str(out.get("fail_reason", "unknown")),
        "command_path": str(out.get("command_path", "unknown")),
        "home_q": home_q.tolist(),
        "q_after": out.get("q_after", []),
        "q_before": out.get("q_before", np.asarray(q_before_full, dtype=float).tolist()),
        "reset_skipped_near_home": False,
        "reset_near_home_l2": near_home_l2,
        "runtime": out,
    }


def _run_episode_gz(
    *,
    ep_id: str,
    ep_index: int,
    step_count: int,
    logs_root: Path,
    runtime: RuntimeROS2Adapter,
    ee_target: np.ndarray,
    controlled_indices: list[int],
    policy_fn: PolicyStepFn,
    action_limit: float,
    reward_config: RewardConfig,
    success_dwell_pos_m: float,
    success_dwell_steps_required: int,
    no_effect_epsilon: float = _NO_EFFECT_EPS,
    no_effect_streak_limit: int = _NO_EFFECT_STREAK_LIMIT,
) -> dict[str, Any]:
    ts0 = time.time_ns()
    executor_cfg = L3ExecutorConfig(dt=0.1, delta_q_limit=(float(action_limit),) * _CONTROLLED_ACTION_DIM)
    executor = L3DeterministicExecutor(executor_cfg)
    action_scale = float(reward_config.action_scale)
    reject_delta_threshold = float(reward_config.reject_delta_threshold)

    l1_path = logs_root / "l1" / f"{ep_id}.jsonl"
    l2_path = logs_root / "l2" / f"{ep_id}.jsonl"
    l3_path = logs_root / "l3" / f"{ep_id}.jsonl"

    prev_q_des: np.ndarray | None = None
    trace_steps: list[dict[str, Any]] = []
    controlled_idx_np = np.asarray(controlled_indices, dtype=int)
    no_effect_streak = 0
    success_dwell_count = 0
    prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
    prev_q: np.ndarray | None = None

    for step in range(max(1, int(step_count))):
        now_ns = ts0 + step * 100_000_000
        q_before_full = runtime.read_q()
        q_before = q_before_full[controlled_idx_np]
        ee_pose_before = _ee_pose_from_q(q_before_full)
        ee_pos_err_prev, ee_ori_err_prev = _ee_errors(ee_pose_before, ee_target)
        goal_error_prev = RewardComposer.ee_error_norm(ee_pos_err_prev, ee_ori_err_prev)
        dq = (q_before - prev_q) if prev_q is not None else np.zeros_like(q_before)
        obs = _obs_from_state(q=q_before, dq=dq, ee_pose_err=np.concatenate([ee_pos_err_prev, ee_ori_err_prev]), prev_action=prev_action)
        policy_out = policy_fn(obs)
        if len(policy_out) == 2:
            action_raw, policy_name = policy_out
            policy_debug: dict[str, Any] = {}
        else:
            action_raw, policy_name, policy_debug = policy_out
        prev_q = q_before.copy()

        l1_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "observation": {"q": q_before.tolist(), "ee_target": ee_target.tolist(), "goal_error_l2": goal_error_prev},
        }
        _jsonl_append(l1_path, l1_payload)

        exec_out = executor.compute_q_des(q_current=q_before, delta_q_cmd=action_raw, prev_q_des=prev_q_des)
        prev_q_des_in = q_before.copy() if prev_q_des is None else np.asarray(prev_q_des, dtype=float).copy()
        saturation = bool(np.any(np.abs(exec_out.clamped_delta_q - exec_out.requested_delta_q) > 1e-12))
        action_exec_nominal = np.asarray(exec_out.q_des - q_before, dtype=float)
        delta_a_nominal = action_exec_nominal - np.asarray(exec_out.requested_delta_q, dtype=float)
        raw_norm = _scaled_action_norm(exec_out.requested_delta_q, action_scale)
        exec_norm_nominal = _scaled_action_norm(action_exec_nominal, action_scale)
        delta_norm_nominal = _scaled_action_norm(delta_a_nominal, action_scale)
        rejected = bool(
            delta_norm_nominal > reject_delta_threshold
            or (
                bool(exec_out.projection_applied)
                and delta_norm_nominal > max(0.4, reject_delta_threshold * 0.5)
            )
        )
        reject_reason = "none"
        q_des_applied = np.asarray(exec_out.q_des, dtype=float)
        if rejected:
            q_des_applied = q_before.copy()
            reject_reason = (
                "projection_heavy"
                if bool(exec_out.projection_applied)
                else "delta_norm_exceeds_threshold"
            )
        prev_q_des = q_des_applied

        l2_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "policy": policy_name,
            "action_raw": exec_out.requested_delta_q.tolist(),
            "action_clamped": exec_out.clamped_delta_q.tolist(),
            "action_exec_nominal": action_exec_nominal.tolist(),
            "delta_a_nominal": delta_a_nominal.tolist(),
            "raw_norm": raw_norm,
            "exec_norm_nominal": exec_norm_nominal,
            "delta_norm_nominal": delta_norm_nominal,
            "clamp_triggered": saturation,
            "projection_applied": bool(exec_out.projection_applied),
            "rejected": rejected,
            "reject_reason": reject_reason,
            "saturated": saturation,
            "policy_debug": policy_debug,
        }
        _jsonl_append(l2_path, l2_payload)

        cmd_q_full = _expand_cmd_q(q_before_full=q_before_full, controlled_indices=controlled_indices, q_des_controlled=q_des_applied)
        rt = runtime.step(cmd_q_full)
        q_after_full = np.asarray(rt["q_after"], dtype=float)
        q_after = q_after_full[controlled_idx_np]
        action_exec = np.asarray(q_after - q_before, dtype=float)
        delta_a = action_exec - np.asarray(exec_out.requested_delta_q, dtype=float)
        exec_norm = _scaled_action_norm(action_exec, action_scale)
        delta_norm = _scaled_action_norm(delta_a, action_scale)
        ee_pose_after = _ee_pose_from_q(q_after_full)
        ee_pos_err_next, ee_ori_err_next = _ee_errors(ee_pose_after, ee_target)
        goal_error_next = RewardComposer.ee_error_norm(ee_pos_err_next, ee_ori_err_next)

        rt_joint_delta_l2 = float(rt["joint_delta_l2"])
        rt_no_effect = rt.get("no_effect")
        no_effect = bool(rt_no_effect) if rt_no_effect is not None else (rt_joint_delta_l2 < float(no_effect_epsilon))
        no_effect_streak = (no_effect_streak + 1) if no_effect else 0
        in_success_dwell = float(np.linalg.norm(ee_pos_err_next)) < float(success_dwell_pos_m)
        success_dwell_count = (success_dwell_count + 1) if in_success_dwell else 0
        success_by_dwell = success_dwell_count >= int(success_dwell_steps_required)

        l3_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "cmd_q": rt["cmd_q"],
            "q_before": rt["q_before"],
            "q_after": rt["q_after"],
            "action_exec": action_exec.tolist(),
            "delta_a": delta_a.tolist(),
            "raw_norm": raw_norm,
            "exec_norm": exec_norm,
            "delta_norm": delta_norm,
            "delta_norm_nominal": delta_norm_nominal,
            "rejected": rejected,
            "reject_reason": reject_reason,
            "joint_delta_l2": rt_joint_delta_l2,
            "goal_error_l2": goal_error_next,
            "no_effect": bool(no_effect),
            "no_effect_streak": int(no_effect_streak),
            "accepted": bool(rt.get("accepted", True)),
            "result_status": rt.get("result_status", "success"),
            "execution_ok": bool(rt.get("execution_ok", True)),
            "fail_reason": rt.get("fail_reason", "none"),
            "success_dwell_count": int(success_dwell_count),
            "success_by_dwell": bool(success_by_dwell),
        }
        _jsonl_append(l3_path, l3_payload)

        intervention = "none"
        if success_by_dwell:
            intervention = "success"
        elif no_effect_streak >= int(no_effect_streak_limit) and float(np.linalg.norm(ee_pos_err_next)) >= 0.08:
            intervention = "no_effect"

        trace_steps.append(
            {
                "step": int(step),
                "obs_q": q_before.tolist(),
                "q_after": q_after.tolist(),
                "ee_pose": ee_pose_before.tolist(),
                "ee_target": ee_target.tolist(),
                "ee_pos_err": ee_pos_err_next.tolist(),
                "ee_ori_err": ee_ori_err_next.tolist(),
                "action_raw": exec_out.requested_delta_q.tolist(),
                "action_clamped": exec_out.clamped_delta_q.tolist(),
                "action_exec_nominal": action_exec_nominal.tolist(),
                "action_exec": action_exec.tolist(),
                "delta_a_nominal": delta_a_nominal.tolist(),
                "delta_a": delta_a.tolist(),
                "raw_norm": raw_norm,
                "exec_norm": exec_norm,
                "exec_norm_nominal": exec_norm_nominal,
                "delta_norm": delta_norm,
                "delta_norm_nominal": delta_norm_nominal,
                "clamp_triggered": saturation,
                "policy_debug": policy_debug,
                "goal_error_prev": goal_error_prev,
                "goal_error_next": goal_error_next,
                "intervention": intervention,
                "projection_applied": bool(exec_out.projection_applied),
                "rejected": rejected,
                "reject_reason": reject_reason,
                "saturated": saturation,
                "no_effect": bool(no_effect),
                "no_effect_streak": int(no_effect_streak),
                "success_dwell_count": int(success_dwell_count),
                "success_by_dwell": bool(success_by_dwell),
                "prev_q_des_in": prev_q_des_in.tolist(),
                "q_des_applied": q_des_applied.tolist(),
                "delta_limits": np.asarray(executor_cfg.delta_q_limit, dtype=float).tolist(),
                "runtime": rt,
            }
        )

        prev_action = action_exec.copy()

        if intervention in {"no_effect", "success"}:
            break

    return {
        "l1": str(l1_path),
        "l2": str(l2_path),
        "l3": str(l3_path),
        "trace_steps": trace_steps,
        "final_goal_error": float(trace_steps[-1]["goal_error_next"]) if trace_steps else 0.0,
    }


def _summarize_gz_episode_trace(
    *,
    trace_steps: list[dict[str, Any]],
    reward_composer: RewardComposer,
    reward_trace: RewardTraceWriter | None,
    reward_trace_kind: str,
    runtime_trace_path: Path | None,
    run_id: str,
    episode: int,
    episode_id: str,
    policy_mode: str,
    runtime_mode: str,
    ee_pos_success_threshold: float,
    ee_ori_success_threshold: float,
    progress_label: str,
    progress_ep_num: int,
    progress_ep_total: int,
) -> dict[str, Any]:
    episode_return = 0.0
    prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
    reward_composer.reset_episode_state()
    ep_component_sums: dict[str, float] = {
        "progress": 0.0,
        "action": 0.0,
        "jerk": 0.0,
        "adjust_penalty": 0.0,
        "raw_action_penalty": 0.0,
        "reject_penalty": 0.0,
        "intervention": 0.0,
        "clamp_or_projection": 0.0,
        "stall": 0.0,
        "ee_small_motion_penalty": 0.0,
        "timeout_penalty": 0.0,
        "reset_fail_penalty": 0.0,
        "execution_fail_penalty": 0.0,
        "timeout_or_reset": 0.0,
        "success_bonus": 0.0,
        "near_goal": 0.0,
        "near_goal_shell": 0.0,
        "inner_shell": 0.0,
        "dwell": 0.0,
        "outer_exit": 0.0,
        "inner_exit": 0.0,
        "zone_exit": 0.0,
        "near_goal_exit": 0.0,
        "local_drift_penalty": 0.0,
        "dwell_break": 0.0,
        "ori_progress": 0.0,
        "reward_total": 0.0,
    }
    ep_max_dwell_count = 0
    ep_dwell_break_count = 0
    ep_clamp_count = 0
    ep_near_goal_entry_count = 0
    ep_near_goal_shell_count = 0
    ep_inner_shell_count = 0
    ep_near_goal_exit_count = 0
    ep_zone_exit_count = 0
    ep_success_triggered_by_dwell = False
    ep_reject_count = 0
    ep_projection_count = 0
    ep_sum_delta_norm = 0.0
    ep_drift_sum = 0.0
    ep_min_dpos = float("inf")
    ep_final_dpos = 0.0
    ep_intervention = 0
    terminal_done = False
    terminal_done_reason = "none"
    terminal_step: int | None = None

    for idx, step in enumerate(trace_steps):
        action_raw = np.asarray(step["action_raw"], dtype=float)
        action_exec = np.asarray(step.get("action_exec", step.get("action_clamped", step["action_raw"])), dtype=float)
        delta_norm = float(step.get("delta_norm", _scaled_action_norm(action_exec - action_raw, reward_composer.config.action_scale)))
        raw_norm = float(step.get("raw_norm", _scaled_action_norm(action_raw, reward_composer.config.action_scale)))
        exec_norm = float(step.get("exec_norm", _scaled_action_norm(action_exec, reward_composer.config.action_scale)))
        step_ee_target = np.asarray(step.get("ee_target", np.zeros(6, dtype=float)), dtype=float)
        step_ee_pose = np.asarray(step.get("ee_pose", np.zeros(6, dtype=float)), dtype=float)
        prev_ee_pos_err = np.asarray(step_ee_target[:3], dtype=float) - np.asarray(step_ee_pose[:3], dtype=float)
        prev_ee_ori_err = np.asarray(step_ee_target[3:6], dtype=float) - np.asarray(step_ee_pose[3:6], dtype=float)
        curr_ee_pos_err = np.asarray(step.get("ee_pos_err", prev_ee_pos_err), dtype=float)
        curr_ee_ori_err = np.asarray(step.get("ee_ori_err", prev_ee_ori_err), dtype=float)
        prev_error = float(step.get("goal_error_prev", RewardComposer.ee_error_norm(prev_ee_pos_err, prev_ee_ori_err)))
        curr_error = float(step.get("goal_error_next", RewardComposer.ee_error_norm(curr_ee_pos_err, curr_ee_ori_err)))
        ee_step_dpos = float(np.linalg.norm(curr_ee_pos_err - prev_ee_pos_err))
        ee_step_dori = float(np.linalg.norm(curr_ee_ori_err - prev_ee_ori_err))
        intervention_now = step["intervention"] != "none"
        clamp_or_projection = bool(step["saturated"] or step["projection_applied"])
        execution_ok = bool(step.get("runtime", {}).get("execution_ok", True))

        done = idx == len(trace_steps) - 1
        done_reason = "running"
        if not execution_ok:
            done = True
            done_reason = "execution_fail"
        elif done:
            if bool(step.get("success_by_dwell", False)):
                done_reason = "success"
            elif step["intervention"] == "no_effect":
                done_reason = "no_effect"
            else:
                done_reason = (
                    "success"
                    if (
                        float(np.linalg.norm(curr_ee_pos_err)) < float(ee_pos_success_threshold)
                        and float(np.linalg.norm(curr_ee_ori_err)) < float(ee_ori_success_threshold)
                    )
                    else "timeout"
                )

        if done:
            terminal_done = True
            terminal_done_reason = done_reason
            terminal_step = int(step["step"])

        q_before_arr = np.asarray(step.get("obs_q", []), dtype=float)
        q_after_arr = np.asarray(step.get("q_after", []), dtype=float)
        q_before_for_stall = q_before_arr if q_before_arr.size > 0 and q_before_arr.shape == q_after_arr.shape else None
        q_after_for_stall = q_after_arr if q_before_for_stall is not None else None

        reward_state_in = reward_composer.state_dict()
        terms = reward_composer.compute(
            prev_ee_pos_err=prev_ee_pos_err,
            prev_ee_ori_err=prev_ee_ori_err,
            curr_ee_pos_err=curr_ee_pos_err,
            curr_ee_ori_err=curr_ee_ori_err,
            action=action_exec,
            action_raw=action_raw,
            action_exec=action_exec,
            prev_action=prev_action,
            intervention=intervention_now,
            clamp_or_projection=clamp_or_projection,
            rejected=bool(step.get("rejected", False)),
            done=done,
            done_reason=done_reason,
            q_before=q_before_for_stall,
            q_after=q_after_for_stall,
            effect_ratio=(step.get("runtime", {}) or {}).get("effect_ratio"),
        )
        reward_state_out = reward_composer.state_dict()
        prev_action = action_exec
        terms_dict = terms.to_dict()
        episode_return += terms.reward_total
        for k in ep_component_sums:
            ep_component_sums[k] += float(terms_dict[k])
        ep_max_dwell_count = max(ep_max_dwell_count, int(terms.dwell_count))
        ep_dwell_break_count += int(terms.dwell_break != 0.0)
        ep_clamp_count += int(clamp_or_projection)
        ep_projection_count += int(bool(step.get("projection_applied", False)))
        ep_reject_count += int(bool(step.get("rejected", False)))
        ep_sum_delta_norm += delta_norm
        ep_near_goal_entry_count += int(terms.near_goal != 0.0)
        ep_near_goal_shell_count += int(terms.in_near_goal_shell > 0.5)
        ep_inner_shell_count += int(terms.in_inner_shell > 0.5)
        ep_near_goal_exit_count += int(terms.near_goal_exit != 0.0)
        ep_zone_exit_count += int(terms.zone_exit != 0.0)
        ep_success_triggered_by_dwell = ep_success_triggered_by_dwell or bool(terms.success_triggered_by_dwell)
        ep_drift_sum += float(terms.local_drift_penalty)
        curr_dpos = float(np.linalg.norm(curr_ee_pos_err))
        ep_min_dpos = min(ep_min_dpos, curr_dpos)
        ep_final_dpos = curr_dpos

        should_log_step = (
            idx == 0
            or done
            or terms.success_triggered_by_dwell > 0.0
            or ((idx + 1) % _PROGRESS_LOG_EVERY_STEPS == 0)
        )
        if should_log_step:
            _progress_log(
                f"{progress_label} ep={progress_ep_num}/{progress_ep_total} step={idx + 1}/{len(trace_steps)} "
                f"dpos={curr_dpos:.4f} reward={terms.reward_total:+.4f} "
                f"dwell={int(terms.dwell_count)} clamp={int(clamp_or_projection)} "
                f"near_entry={int(terms.near_goal != 0.0)} "
                f"delta_norm={delta_norm:.3f} reject={int(bool(step.get('rejected', False)))} "
                f"sat_dims={int((step.get('policy_debug') or {}).get('saturated_dims', 0))} "
                f"pre_tanh_max={float((step.get('policy_debug') or {}).get('pre_tanh_abs_max', 0.0)):.3f} "
                f"done={done_reason}"
            )

        if reward_trace is not None:
            reward_trace.append(
                {
                    "run_id": run_id,
                    "trace_kind": reward_trace_kind,
                    "episode": episode,
                    "episode_id": episode_id,
                    "step": int(step["step"]),
                    "policy_mode": policy_mode,
                    "runtime_mode": runtime_mode,
                    "done": done,
                    "done_reason": done_reason,
                    "goal_error_prev": prev_error,
                    "goal_error_next": curr_error,
                    "ee_pose": step.get("ee_pose"),
                    "ee_target": step.get("ee_target"),
                    "ee_pos_err": step.get("ee_pos_err"),
                    "ee_ori_err": step.get("ee_ori_err"),
                    "action_raw": action_raw.tolist(),
                    "action_exec": action_exec.tolist(),
                    "delta_a": (action_exec - action_raw).tolist(),
                    "raw_norm": raw_norm,
                    "exec_norm": exec_norm,
                    "delta_norm": delta_norm,
                    "clamp_triggered": bool(step.get("clamp_triggered", step.get("saturated", False))),
                    "projection_triggered": bool(step.get("projection_applied", False)),
                    "rejected": bool(step.get("rejected", False)),
                    "reject_reason": step.get("reject_reason", "none"),
                    "ee_step_dpos": ee_step_dpos,
                    "ee_step_dori": ee_step_dori,
                    "ee_small_motion_penalty": float(terms.ee_small_motion_penalty),
                    "reward_total": float(terms.reward_total),
                    "policy_debug": step.get("policy_debug", {}),
                    "reward_state_in": reward_state_in,
                    "reward_state_out": reward_state_out,
                    "components": terms_dict,
                }
            )

        if runtime_trace_path is not None:
            _jsonl_append(
                runtime_trace_path,
                {
                    "run_id": run_id,
                    "trace_kind": reward_trace_kind,
                    "episode": episode,
                    "step": int(step["step"]),
                    "cmd_q": step["runtime"]["cmd_q"],
                    "readback_q_before": step["runtime"]["q_before"],
                    "readback_q_after": step["runtime"]["q_after"],
                    "joint_delta": step["runtime"]["joint_delta"],
                    "joint_delta_l2": step["runtime"]["joint_delta_l2"],
                    "cmd_delta_l2": step["runtime"].get("cmd_delta_l2"),
                    "effect_ratio": step["runtime"].get("effect_ratio"),
                    "frame_before_stamp_ns": step["runtime"].get("frame_before_stamp_ns"),
                    "frame_after_stamp_ns": step["runtime"].get("frame_after_stamp_ns"),
                    "goal_error_prev": prev_error,
                    "goal_error_next": curr_error,
                    "ee_pose": step.get("ee_pose"),
                    "ee_target": step.get("ee_target"),
                    "ee_pos_err": step.get("ee_pos_err"),
                    "ee_ori_err": step.get("ee_ori_err"),
                    "action_raw": action_raw.tolist(),
                    "action_exec": action_exec.tolist(),
                    "delta_a": (action_exec - action_raw).tolist(),
                    "raw_norm": raw_norm,
                    "exec_norm": exec_norm,
                    "delta_norm": delta_norm,
                    "clamp_triggered": bool(step.get("clamp_triggered", step.get("saturated", False))),
                    "projection_triggered": bool(step.get("projection_applied", False)),
                    "rejected": bool(step.get("rejected", False)),
                    "reject_reason": step.get("reject_reason", "none"),
                    "policy_debug": step.get("policy_debug", {}),
                    "no_effect": bool(step.get("no_effect", False)),
                    "no_effect_reason": step["runtime"].get("no_effect_reason", "none"),
                    "no_effect_streak": int(step.get("no_effect_streak", 0)),
                    "skipped_publish": bool(step["runtime"].get("skipped_publish", False)),
                    "accepted": bool(step["runtime"].get("accepted", True)),
                    "result_status": step["runtime"].get("result_status", "success"),
                    "execution_ok": bool(step["runtime"].get("execution_ok", True)),
                    "fail_reason": step["runtime"].get("fail_reason", "none"),
                    "command_path": step["runtime"].get("command_path", "topic_fallback"),
                    "action_status": step["runtime"].get("action_status"),
                    "action_error_code": step["runtime"].get("action_error_code"),
                    "timestamp_ns": step["runtime"]["timestamp_ns"],
                },
            )

        if intervention_now:
            ep_intervention = 1

        if done_reason == "execution_fail":
            ep_intervention = 1
            break

    final_error = float(trace_steps[-1]["goal_error_next"]) if trace_steps else 1.0
    if trace_steps:
        last = trace_steps[-1]
        pos_ok = float(np.linalg.norm(np.asarray(last.get("ee_pos_err", [9, 9, 9]), dtype=float))) < float(ee_pos_success_threshold)
        ori_ok = float(np.linalg.norm(np.asarray(last.get("ee_ori_err", [9, 9, 9]), dtype=float))) < float(ee_ori_success_threshold)
        ep_success = 1 if (bool(last.get("success_by_dwell", False)) or (pos_ok and ori_ok)) else 0
    else:
        ep_success = 0
    zone_metrics = _dpos_zone_metrics(
        min_dpos=float(ep_min_dpos if trace_steps else 0.0),
        final_dpos=float(ep_final_dpos if trace_steps else 0.0),
        reward_config=reward_composer.config,
    )

    return {
        "episode_return": float(episode_return),
        "component_sums": ep_component_sums,
        "max_dwell_count": int(ep_max_dwell_count),
        "dwell_break_count": int(ep_dwell_break_count),
        "clamp_count": int(ep_clamp_count),
        "projection_count": int(ep_projection_count),
        "reject_count": int(ep_reject_count),
        "reject_rate": _safe_rate(ep_reject_count, len(trace_steps)),
        "sum_delta_norm": float(ep_sum_delta_norm),
        "near_goal_entry_count": int(ep_near_goal_entry_count),
        "near_goal_shell_count": int(ep_near_goal_shell_count),
        "inner_shell_count": int(ep_inner_shell_count),
        "near_goal_exit_count": int(ep_near_goal_exit_count),
        "zone_exit_count": int(ep_zone_exit_count),
        "shell_hit": bool(ep_near_goal_shell_count > 0),
        "inner_shell_hit": bool(ep_inner_shell_count > 0),
        "dwell_hit": bool(ep_max_dwell_count > 0),
        "drift_sum": float(ep_drift_sum),
        "success_triggered_by_dwell": bool(ep_success_triggered_by_dwell),
        "min_dpos": float(ep_min_dpos if trace_steps else 0.0),
        "final_dpos": float(ep_final_dpos if trace_steps else 0.0),
        "intervention_count": int(ep_intervention),
        "terminal": bool(terminal_done),
        "done_reason": terminal_done_reason,
        "terminal_step": terminal_step,
        "success_count": int(ep_success),
        "final_goal_error": final_error,
        "step_count": max(1, len(trace_steps)),
        **zone_metrics,
    }


def _run_post_training_eval_gz(
    *,
    run_id: str,
    artifact_root: Path,
    eval_root: Path | None,
    agent: Any,
    runtime: RuntimeROS2Adapter,
    stage: Any,
    target_mode: str,
    near_home_profile: str,
    near_home_pos_offset_min_m: float,
    near_home_pos_offset_max_m: float,
    near_home_ori_offset_min_deg: float,
    near_home_ori_offset_max_deg: float,
    external_ee_target: np.ndarray,
    external_ee_target_source: dict[str, Any],
    runtime_controlled_indices: list[int],
    policy_mode: str,
    runtime_mode: str,
    episodes: int,
    steps_per_episode: int,
    ee_pos_success_threshold: float,
    ee_ori_success_threshold: float,
    reset_near_home_eps: float,
    sac_seed: int,
    gz_world_name: str,
    gz_target_entity_name: str,
    gz_target_world_offset_xyz: tuple[float, float, float],
    reward_config: RewardConfig,
    fixed_eval_suite: dict[str, Any] | None = None,
    eval_name: str = "deterministic",
    eval_stochastic: bool = False,
    eval_exploration_std_scale: float = 0.0,
    artifact_key_prefix: str = "post_train_eval",
) -> dict[str, Any]:
    eval_root = Path(eval_root) if eval_root is not None else (artifact_root / "eval")
    eval_logs_root = eval_root / "logs"
    eval_logs_root.mkdir(parents=True, exist_ok=True)
    eval_slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(eval_name).strip().lower())
    eval_slug = eval_slug or "eval"
    eval_reward_trace_path = eval_root / f"{eval_slug}_reward_trace.jsonl"
    eval_runtime_trace_path = eval_root / f"{eval_slug}_runtime_trace.jsonl"
    eval_episode_summary_path = eval_root / f"{eval_slug}_episode_summary.jsonl"
    eval_summary_path = eval_root / f"{eval_slug}_summary.json"
    eval_policy_suffix = eval_slug

    eval_reward_trace = RewardTraceWriter(eval_reward_trace_path)
    eval_runtime_trace_path.write_text("", encoding="utf-8")
    eval_episode_summary_path.write_text("", encoding="utf-8")
    reward_composer = RewardComposer(reward_config)
    resolved_target_mode = target_mode
    if resolved_target_mode == "auto":
        resolved_target_mode = "near_home" if stage.name == "S0_B" else "external"
    if fixed_eval_suite is not None:
        resolved_target_mode = str(fixed_eval_suite.get("resolved_target_mode", resolved_target_mode))

    _progress_log(
        f"eval_start run_id={run_id} episodes={episodes} steps_per_episode={steps_per_episode} "
        f"target_mode={resolved_target_mode} policy_mode={policy_mode}_{eval_policy_suffix} "
        f"stochastic={bool(eval_stochastic)} exploration_std_scale={float(eval_exploration_std_scale):.3f}"
    )

    episode_outputs: list[dict[str, Any]] = []
    successes = 0
    interventions = 0

    for ep in range(max(0, int(episodes))):
        ep_id = f"{run_id}_eval_ep{ep:03d}_{stage.name}"
        step_count = min(int(steps_per_episode), int(stage.step_budget))
        _progress_log(
            f"eval_episode_start ep={ep + 1}/{episodes} ep_id={ep_id} "
            f"stage={stage.name} step_budget={step_count} target_mode={resolved_target_mode}"
        )

        reset_info = _reset_episode_home(
            runtime=runtime,
            controlled_indices=runtime_controlled_indices,
            home_q=_HOME_Q,
            reset_near_home_eps=reset_near_home_eps,
        )
        _jsonl_append(
            eval_runtime_trace_path,
            {"run_id": run_id, "trace_kind": "post_train_eval", "episode": ep, "step": -1, "event": "reset", **reset_info},
        )
        if not (reset_info["accepted"] and reset_info["execution_ok"]):
            _progress_log(
                f"eval_reset_fail ep={ep + 1}/{episodes} ep_id={ep_id} reason={reset_info.get('fail_reason', 'unknown')}"
            )
            break

        q_after_reset = np.asarray(reset_info.get("q_after", _HOME_Q.tolist()), dtype=float)
        home_q_after_reset = q_after_reset[np.asarray(runtime_controlled_indices, dtype=int)]
        fixed_targets = list((fixed_eval_suite or {}).get("targets", []))
        if fixed_targets:
            target_spec = fixed_targets[ep % len(fixed_targets)]
            ee_target = np.asarray(target_spec.get("ee_target", external_ee_target), dtype=float).copy()
            ee_target_source = dict(target_spec.get("ee_target_source", external_ee_target_source))
        elif resolved_target_mode == "near_home":
            ee_target, ee_target_source = _resolve_near_home_ee_target(
                home_q=home_q_after_reset,
                profile=near_home_profile,
                pos_offset_min_m=near_home_pos_offset_min_m,
                pos_offset_max_m=near_home_pos_offset_max_m,
                ori_offset_min_deg=near_home_ori_offset_min_deg,
                ori_offset_max_deg=near_home_ori_offset_max_deg,
                rng=np.random.default_rng(int(sac_seed) + 100_003 + int(ep)),
            )
        else:
            ee_target = np.asarray(external_ee_target, dtype=float).copy()
            ee_target_source = dict(external_ee_target_source)

        target_visualization = {
            "success": False,
            "action": "disabled",
            "reason": "not_attempted",
            "world_name": gz_world_name,
            "entity_name": gz_target_entity_name,
            "pose_offset_xyz": [float(v) for v in gz_target_world_offset_xyz],
        }
        publish_target_visual = getattr(runtime, "publish_ee_target_visual", None)
        if callable(publish_target_visual):
            target_visualization = dict(publish_target_visual(ee_target))
            if bool(target_visualization.get("success", False)):
                _progress_log(
                    f"eval_target_visual ep={ep + 1}/{episodes} ep_id={ep_id} "
                    f"action={target_visualization.get('action', 'unknown')} "
                    f"world={target_visualization.get('world_name', gz_world_name)} "
                    f"entity={target_visualization.get('entity_name', gz_target_entity_name)}"
                )
            else:
                _progress_log(
                    f"eval_target_visual_warn ep={ep + 1}/{episodes} ep_id={ep_id} "
                    f"reason={target_visualization.get('reason', 'unknown')}"
                )

        def _eval_policy_fn(obs: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
            action, diagnostics = agent.act_with_diagnostics(
                obs,
                stochastic=bool(eval_stochastic),
                exploration_std_scale=float(eval_exploration_std_scale),
            )
            return action, f"{policy_mode}_{eval_policy_suffix}", diagnostics

        logs = _run_episode_gz(
            ep_id=ep_id,
            ep_index=ep,
            step_count=step_count,
            logs_root=eval_logs_root,
            runtime=runtime,
            ee_target=ee_target,
            controlled_indices=runtime_controlled_indices,
            policy_fn=_eval_policy_fn,
            action_limit=float(stage.action_limit),
            reward_config=reward_composer.config,
            success_dwell_pos_m=float(reward_composer.config.dwell_pos_m),
            success_dwell_steps_required=int(reward_composer.config.dwell_steps_required),
        )

        trace_summary = _summarize_gz_episode_trace(
            trace_steps=logs.get("trace_steps", []),
            reward_composer=reward_composer,
            reward_trace=eval_reward_trace,
            reward_trace_kind="post_train_eval",
            runtime_trace_path=eval_runtime_trace_path,
            run_id=run_id,
            episode=ep,
            episode_id=ep_id,
            policy_mode=f"{policy_mode}_{eval_policy_suffix}",
            runtime_mode=runtime_mode,
            ee_pos_success_threshold=ee_pos_success_threshold,
            ee_ori_success_threshold=ee_ori_success_threshold,
            progress_label="eval_step",
            progress_ep_num=ep + 1,
            progress_ep_total=max(1, int(episodes)),
        )

        step_n = int(trace_summary["step_count"])
        ep_summary = {
            "run_id": run_id,
            "episode": ep,
            "episode_id": ep_id,
            "stage": stage.name,
            "component_sums": trace_summary["component_sums"],
            "component_means": {k: float(v) / float(step_n) for k, v in trace_summary["component_sums"].items()},
            "total_reward": float(trace_summary["component_sums"]["reward_total"]),
            "success_count": int(trace_summary["success_count"]),
            "intervention_count": int(trace_summary["intervention_count"]),
            "target_mode": resolved_target_mode,
            "home_ee": ee_target_source.get("home_ee"),
            "ee_target": ee_target.tolist(),
            "target_delta_pos": ee_target_source.get("target_delta_pos"),
            "target_delta_ori": ee_target_source.get("target_delta_ori"),
            "target_visualization": target_visualization,
            "max_dwell_count": int(trace_summary["max_dwell_count"]),
            "dwell_break_count": int(trace_summary["dwell_break_count"]),
            "clamp_count": int(trace_summary["clamp_count"]),
            "projection_count": int(trace_summary["projection_count"]),
            "reject_count": int(trace_summary["reject_count"]),
            "reject_rate": float(trace_summary["reject_rate"]),
            "sum_delta_norm": float(trace_summary["sum_delta_norm"]),
            "near_goal_entry_count": int(trace_summary["near_goal_entry_count"]),
            "near_goal_shell_count": int(trace_summary["near_goal_shell_count"]),
            "inner_shell_count": int(trace_summary["inner_shell_count"]),
            "near_goal_exit_count": int(trace_summary["near_goal_exit_count"]),
            "zone_exit_count": int(trace_summary["zone_exit_count"]),
            "shell_hit": bool(trace_summary["shell_hit"]),
            "inner_shell_hit": bool(trace_summary["inner_shell_hit"]),
            "dwell_hit": bool(trace_summary["dwell_hit"]),
            "drift_sum": float(trace_summary["drift_sum"]),
            "success_triggered_by_dwell": bool(trace_summary["success_triggered_by_dwell"]),
            "min_dpos": float(trace_summary["min_dpos"]),
            "final_dpos": float(trace_summary["final_dpos"]),
            "final_dpos_minus_min_dpos": float(trace_summary["final_dpos"] - trace_summary["min_dpos"]),
            "true_min_zone": trace_summary["true_min_zone"],
            "true_final_zone": trace_summary["true_final_zone"],
            "true_outer_hit": bool(trace_summary["true_outer_hit"]),
            "true_inner_hit": bool(trace_summary["true_inner_hit"]),
            "true_dwell_hit": bool(trace_summary["true_dwell_hit"]),
            "true_basin_hit": bool(trace_summary["true_basin_hit"]),
            "true_inner_or_dwell_hit": bool(trace_summary["true_inner_or_dwell_hit"]),
            "true_final_outer": bool(trace_summary["true_final_outer"]),
            "true_final_inner": bool(trace_summary["true_final_inner"]),
            "true_final_dwell": bool(trace_summary["true_final_dwell"]),
            "true_final_basin": bool(trace_summary["true_final_basin"]),
            "true_final_inner_or_dwell": bool(trace_summary["true_final_inner_or_dwell"]),
            "reset_result": reset_info,
            "terminal": bool(trace_summary["terminal"]),
            "done_reason": trace_summary["done_reason"],
            "terminal_step": trace_summary["terminal_step"],
            "policy_mode": f"{policy_mode}_{eval_policy_suffix}",
            "eval_name": eval_slug,
            "eval_stochastic": bool(eval_stochastic),
            "eval_exploration_std_scale": float(eval_exploration_std_scale),
            "runtime_mode": runtime_mode,
        }
        _jsonl_append(eval_episode_summary_path, ep_summary)

        _progress_log(
            f"eval_episode_end ep={ep + 1}/{episodes} ep_id={ep_id} success={trace_summary['success_count']} "
            f"done_reason={trace_summary['done_reason']} return={trace_summary['episode_return']:+.4f} "
            f"min_dpos={trace_summary['min_dpos']:.4f} final_dpos={trace_summary['final_dpos']:.4f} "
            f"max_dwell={trace_summary['max_dwell_count']} clamp_count={trace_summary['clamp_count']}"
        )

        episode_outputs.append(ep_summary)
        successes += int(trace_summary["success_count"])
        interventions += int(trace_summary["intervention_count"])

    min_dpos_values = [float(ep.get("min_dpos", 0.0)) for ep in episode_outputs]
    final_dpos_values = [float(ep.get("final_dpos", 0.0)) for ep in episode_outputs]
    clamp_values = [float(ep.get("clamp_count", 0.0)) for ep in episode_outputs]
    reject_values = [float(ep.get("reject_count", 0.0)) for ep in episode_outputs]
    near_entry_values = [float(ep.get("near_goal_entry_count", 0.0)) for ep in episode_outputs]
    shell_hit_values = [1.0 if bool(ep.get("shell_hit", False)) else 0.0 for ep in episode_outputs]
    inner_shell_hit_values = [1.0 if bool(ep.get("inner_shell_hit", False)) else 0.0 for ep in episode_outputs]
    dwell_hit_values = [1.0 if bool(ep.get("dwell_hit", False)) else 0.0 for ep in episode_outputs]
    true_outer_hit_values = [1.0 if bool(ep.get("true_outer_hit", False)) else 0.0 for ep in episode_outputs]
    true_inner_hit_values = [1.0 if bool(ep.get("true_inner_hit", False)) else 0.0 for ep in episode_outputs]
    true_dwell_hit_values = [1.0 if bool(ep.get("true_dwell_hit", False)) else 0.0 for ep in episode_outputs]
    true_basin_hit_values = [1.0 if bool(ep.get("true_basin_hit", False)) else 0.0 for ep in episode_outputs]
    true_final_outer_values = [1.0 if bool(ep.get("true_final_outer", False)) else 0.0 for ep in episode_outputs]
    true_final_inner_values = [1.0 if bool(ep.get("true_final_inner", False)) else 0.0 for ep in episode_outputs]
    true_final_dwell_values = [1.0 if bool(ep.get("true_final_dwell", False)) else 0.0 for ep in episode_outputs]
    true_final_basin_values = [1.0 if bool(ep.get("true_final_basin", False)) else 0.0 for ep in episode_outputs]
    final_minus_min_values = [float(ep.get("final_dpos_minus_min_dpos", 0.0)) for ep in episode_outputs]
    reward_values = [float(ep.get("total_reward", 0.0)) for ep in episode_outputs]
    regression_count = sum(1 for ep in episode_outputs if float(ep.get("final_dpos", 0.0)) > float(ep.get("min_dpos", 0.0)))

    metrics = {
        "episodes_requested": int(episodes),
        "episodes_completed": int(len(episode_outputs)),
        "success_rate": _safe_rate(successes, len(episode_outputs)),
        "intervention_rate": _safe_rate(interventions, len(episode_outputs)),
        "reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "reward_min": float(np.min(reward_values)) if reward_values else 0.0,
        "reward_max": float(np.max(reward_values)) if reward_values else 0.0,
        "best_min_dpos": float(np.min(min_dpos_values)) if min_dpos_values else 0.0,
        "mean_min_dpos": float(np.mean(min_dpos_values)) if min_dpos_values else 0.0,
        "mean_final_dpos": float(np.mean(final_dpos_values)) if final_dpos_values else 0.0,
        "clamp_sum": float(np.sum(clamp_values)) if clamp_values else 0.0,
        "clamp_mean": float(np.mean(clamp_values)) if clamp_values else 0.0,
        "reject_sum": float(np.sum(reject_values)) if reject_values else 0.0,
        "reject_mean": float(np.mean(reject_values)) if reject_values else 0.0,
        "near_goal_entries_sum": float(np.sum(near_entry_values)) if near_entry_values else 0.0,
        "shell_hit_rate": float(np.mean(shell_hit_values)) if shell_hit_values else 0.0,
        "inner_shell_hit_rate": float(np.mean(inner_shell_hit_values)) if inner_shell_hit_values else 0.0,
        "dwell_hit_rate": float(np.mean(dwell_hit_values)) if dwell_hit_values else 0.0,
        "true_outer_hit_rate": float(np.mean(true_outer_hit_values)) if true_outer_hit_values else 0.0,
        "true_inner_hit_rate": float(np.mean(true_inner_hit_values)) if true_inner_hit_values else 0.0,
        "true_dwell_hit_rate": float(np.mean(true_dwell_hit_values)) if true_dwell_hit_values else 0.0,
        "true_basin_hit_rate": float(np.mean(true_basin_hit_values)) if true_basin_hit_values else 0.0,
        "true_final_outer_rate": float(np.mean(true_final_outer_values)) if true_final_outer_values else 0.0,
        "true_final_inner_rate": float(np.mean(true_final_inner_values)) if true_final_inner_values else 0.0,
        "true_final_dwell_rate": float(np.mean(true_final_dwell_values)) if true_final_dwell_values else 0.0,
        "true_final_basin_rate": float(np.mean(true_final_basin_values)) if true_final_basin_values else 0.0,
        "max_dwell_max": float(max((int(ep.get("max_dwell_count", 0)) for ep in episode_outputs), default=0)),
        "regression_rate": _safe_rate(regression_count, len(episode_outputs)),
        "mean_final_minus_min": float(np.mean(final_minus_min_values)) if final_minus_min_values else 0.0,
    }
    action_stats = _reward_trace_action_stats(eval_reward_trace_path)
    for key in (
        "final_action_l2_mean",
        "raw_norm_mean",
        "exec_norm_mean",
        "delta_norm_mean",
        "mu_l2_mean",
        "std_scaled_l2_mean",
        "noise_l2_mean",
        "pre_tanh_abs_max_mean",
        "pre_tanh_abs_max_max",
    ):
        if key in action_stats:
            metrics[key] = float(action_stats[key])

    summary = {
        "run_id": run_id,
        "mode": f"{eval_slug}_post_train_eval",
        "timestamp_ns": time.time_ns(),
        "policy_mode": f"{policy_mode}_{eval_policy_suffix}",
        "eval_name": eval_slug,
        "eval_stochastic": bool(eval_stochastic),
        "eval_exploration_std_scale": float(eval_exploration_std_scale),
        "runtime_mode": runtime_mode,
        "target_mode": target_mode,
        "resolved_target_mode": resolved_target_mode,
        "stage": stage.name,
        "fixed_eval_suite": dict(fixed_eval_suite or {}),
        "metrics": metrics,
        "action_stats": action_stats,
        "episodes": episode_outputs,
        "artifacts": {
            "logs_root": str(eval_logs_root),
            "reward_trace": str(eval_reward_trace_path),
            "runtime_trace": str(eval_runtime_trace_path),
            "episode_summary": str(eval_episode_summary_path),
            "summary": str(eval_summary_path),
        },
    }
    eval_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _progress_log(
        f"eval_end run_id={run_id} episodes_completed={len(episode_outputs)}/{episodes} "
        f"name={eval_slug} success_rate={metrics['success_rate']:.3f} best_min_dpos={metrics['best_min_dpos']:.4f}"
    )

    artifact_prefix = str(artifact_key_prefix or "post_train_eval")
    return {
        "summary": summary,
        "artifacts": {
            f"{artifact_prefix}_summary": str(eval_summary_path),
            f"{artifact_prefix}_episode_summary": str(eval_episode_summary_path),
            f"{artifact_prefix}_reward_trace": str(eval_reward_trace_path),
            f"{artifact_prefix}_runtime_trace": str(eval_runtime_trace_path),
            f"{artifact_prefix}_logs_root": str(eval_logs_root),
        },
    }


def _run_gap_diagnosis_gz(
    *,
    run_id: str,
    artifact_root: Path,
    agent: Any,
    runtime: RuntimeROS2Adapter,
    stage: Any,
    target_mode: str,
    near_home_profile: str,
    near_home_pos_offset_min_m: float,
    near_home_pos_offset_max_m: float,
    near_home_ori_offset_min_deg: float,
    near_home_ori_offset_max_deg: float,
    external_ee_target: np.ndarray,
    external_ee_target_source: dict[str, Any],
    runtime_controlled_indices: list[int],
    policy_mode: str,
    runtime_mode: str,
    episodes: int,
    steps_per_episode: int,
    ee_pos_success_threshold: float,
    ee_ori_success_threshold: float,
    reset_near_home_eps: float,
    sac_seed: int,
    gz_world_name: str,
    gz_target_entity_name: str,
    gz_target_world_offset_xyz: tuple[float, float, float],
    reward_config: RewardConfig,
    fixed_eval_suite: dict[str, Any],
    eval_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    gap_root = artifact_root / "eval_gap"
    gap_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    for spec in eval_specs:
        label = str(spec["label"])
        eval_root = gap_root / label
        eval_out = _run_post_training_eval_gz(
            run_id=f"{run_id}_gap_{label}",
            artifact_root=artifact_root,
            eval_root=eval_root,
            agent=agent,
            runtime=runtime,
            stage=stage,
            target_mode=target_mode,
            near_home_profile=near_home_profile,
            near_home_pos_offset_min_m=near_home_pos_offset_min_m,
            near_home_pos_offset_max_m=near_home_pos_offset_max_m,
            near_home_ori_offset_min_deg=near_home_ori_offset_min_deg,
            near_home_ori_offset_max_deg=near_home_ori_offset_max_deg,
            external_ee_target=external_ee_target,
            external_ee_target_source=external_ee_target_source,
            runtime_controlled_indices=runtime_controlled_indices,
            policy_mode=policy_mode,
            runtime_mode=runtime_mode,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            ee_pos_success_threshold=ee_pos_success_threshold,
            ee_ori_success_threshold=ee_ori_success_threshold,
            reset_near_home_eps=reset_near_home_eps,
            sac_seed=sac_seed,
            gz_world_name=gz_world_name,
            gz_target_entity_name=gz_target_entity_name,
            gz_target_world_offset_xyz=gz_target_world_offset_xyz,
            reward_config=reward_config,
            fixed_eval_suite=fixed_eval_suite,
            eval_name=label,
            eval_stochastic=bool(spec["stochastic"]),
            eval_exploration_std_scale=float(spec["exploration_std_scale"]),
            artifact_key_prefix=f"gap_eval_{label}",
        )
        metrics = dict(eval_out["summary"].get("metrics", {}))
        action_stats = _reward_trace_action_stats(Path(eval_out["artifacts"][f"gap_eval_{label}_reward_trace"]))
        record = {
            "label": label,
            "stochastic": bool(spec["stochastic"]),
            "exploration_std_scale": float(spec["exploration_std_scale"]),
            "metrics": metrics,
            "action_stats": action_stats,
            "artifacts": dict(eval_out["artifacts"]),
        }
        records.append(record)
        artifacts.update(eval_out["artifacts"])

    deterministic = next((r for r in records if not bool(r["stochastic"])), None)
    full_noise = records[-1] if records else None
    gap_metrics: dict[str, float] = {}
    if deterministic is not None and full_noise is not None:
        det_metrics = dict(deterministic.get("metrics", {}))
        full_metrics = dict(full_noise.get("metrics", {}))
        det_actions = dict(deterministic.get("action_stats", {}))
        full_actions = dict(full_noise.get("action_stats", {}))
        gap_metrics = {
            "success_rate_gap_full_minus_det": float(full_metrics.get("success_rate", 0.0))
            - float(det_metrics.get("success_rate", 0.0)),
            "true_basin_hit_rate_gap_full_minus_det": float(full_metrics.get("true_basin_hit_rate", 0.0))
            - float(det_metrics.get("true_basin_hit_rate", 0.0)),
            "mean_final_dpos_gap_det_minus_full": float(det_metrics.get("mean_final_dpos", 0.0))
            - float(full_metrics.get("mean_final_dpos", 0.0)),
            "final_action_l2_ratio_det_over_full": float(det_actions.get("final_action_l2_mean", 0.0))
            / max(float(full_actions.get("final_action_l2_mean", 0.0)), 1e-8),
            "raw_norm_ratio_det_over_full": float(det_actions.get("raw_norm_mean", 0.0))
            / max(float(full_actions.get("raw_norm_mean", 0.0)), 1e-8),
        }

    summary = {
        "run_id": run_id,
        "mode": "stochastic_to_deterministic_gap_diagnosis",
        "timestamp_ns": time.time_ns(),
        "episodes": int(episodes),
        "steps_per_episode": int(steps_per_episode),
        "stage": str(stage.name),
        "fixed_eval_suite": dict(fixed_eval_suite),
        "records": records,
        "gap_metrics": gap_metrics,
    }
    summary_path = gap_root / "gap_diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["gap_eval_summary"] = str(summary_path)
    artifacts["gap_eval_root"] = str(gap_root)
    return {"summary": summary, "artifacts": artifacts}


def run_pipeline_e2e(
    run_id: str,
    episodes: int,
    steps_per_episode: int,
    artifact_root: Path,
    enforce_gates: bool = False,
    policy_mode: str = "sac_torch",
    sac_seed: int = 0,
    stage_profile: str = "default",
    runtime_mode: str = "smoke",
    runtime_factory: RuntimeFactory | None = None,
    runtime_joint_names: list[str] | None = None,
    trajectory_topic: str = "/arm_controller/joint_trajectory",
    joint_state_topic: str = "/joint_states",
    ee_pos_success_threshold: float = 0.08,
    ee_ori_success_threshold: float = 0.12,
    reset_near_home_eps: float = 1e-4,
    external_task_prop: str = "tray",
    external_task_src_idx: int = 2,
    external_task_dst_idx: int = 7,
    external_task_waypoint_index: int = 2,
    target_mode: str = "auto",
    near_home_profile: str = "s0_bootstrap",
    near_home_pos_offset_min_m: float = 0.22,
    near_home_pos_offset_max_m: float = 0.30,
    near_home_ori_offset_min_deg: float = 5.0,
    near_home_ori_offset_max_deg: float = 10.0,
    gz_visualize_target: bool = True,
    gz_world_name: str = "empty",
    gz_target_entity_name: str = "v5_1_target_marker",
    gz_target_world_offset_x: float = 0.0,
    gz_target_world_offset_y: float = 0.0,
    gz_target_world_offset_z: float = 1.04,
    exploration_std_scale: float = 1.0,
    action_scale: float = 0.05,
    sac_target_entropy: float | None = None,
    reward_profile: str = "default",
    actor_mu_limit: float = 1.5,
    actor_update_delay: int = 2,
    actor_bc_lambda: float = 0.05,
    actor_bc_outer_dpos_m: float = 0.08,
    actor_bc_inner_dpos_m: float = 0.04,
    actor_bc_topk: int = 3,
    actor_distill_lambda: float = 0.0,
    actor_distill_interval: int = 20,
    actor_distill_steps: int = 1,
    actor_distill_batch_size: int = 0,
    actor_distill_candidate_multiplier: int = 8,
    actor_distill_min_good_count: int = 8,
    actor_distill_outer_dpos_m: float = 0.08,
    actor_distill_support_dpos_m: float = 0.07,
    actor_distill_inner_dpos_m: float = 0.04,
    actor_distill_dwell_dpos_m: float = 0.025,
    actor_distill_min_progress_m: float = 0.003,
    actor_distill_max_delta_norm: float = 0.75,
    actor_distill_quality_threshold: float = 0.0,
    actor_distill_advantage_beta: float = 0.0,
    actor_distill_advantage_clip: float = 5.0,
    actor_distill_grad_clip: float = 1.0,
    actor_distill_exclude_rejected: bool = True,
    actor_distill_exclude_clamped: bool = True,
    actor_distill_exclude_projected: bool = True,
    post_train_eval_episodes: int = 5,
    post_train_eval_steps_per_episode: int | None = None,
    gap_eval_scales: str = "",
    gap_eval_episodes: int = 0,
    gap_eval_steps_per_episode: int | None = None,
    entropy_anneal_mode: str = "off",
    entropy_anneal_ratios: str = "1.0,0.70,0.50",
    entropy_anneal_stage_names: str = "A,B,C",
    entropy_anneal_min_episode: int = 20,
    entropy_anneal_window: int = 20,
    entropy_anneal_fixed_episodes: str = "20,40",
    entropy_anneal_max_stage_index: int | None = None,
    distill_start_entropy_stage_index: int = 1,
    periodic_eval_interval: int = 10,
    periodic_eval_episodes: int = 5,
    early_stop_patience_evals: int = 5,
    action_curriculum_max_stage: int | None = None,
    target_curriculum_max_stage: int | None = None,
    disable_exploration_schedule: bool = False,
    resume_best_patience_evals: int = 3,
    max_best_resume_count: int = 0,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    checkpoint_layout = _checkpoint_layout(artifact_root)

    reward_trace_path = artifact_root / "reward_trace.jsonl"
    reward_trace = RewardTraceWriter(reward_trace_path)
    reward_config = _reward_config_for_profile(reward_profile, action_scale=float(action_scale))
    reward_composer = RewardComposer(reward_config)
    target_rng = np.random.default_rng(int(sac_seed))
    episode_reward_summary_path = artifact_root / "episode_reward_summary.jsonl"
    episode_reward_summary_path.write_text("", encoding="utf-8")
    train_metrics_path = artifact_root / "train_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    eval_step_budget = (
        int(post_train_eval_steps_per_episode)
        if post_train_eval_steps_per_episode is not None
        else int(steps_per_episode)
    )

    runtime_trace_path = artifact_root / "runtime_trace.jsonl"
    runtime_trace_path.write_text("", encoding="utf-8")

    curriculum = CurriculumManager(stages=resolve_stages(stage_profile), max_stage_index=action_curriculum_max_stage)
    gate_eval = GateEvaluator(DEFAULT_GATE)

    if policy_mode != "sac_torch":
        raise ValueError("V5.1 single-path only supports policy_mode=sac_torch")
    if runtime_mode not in {"smoke", "gz"}:
        raise ValueError("runtime_mode must be one of: smoke|gz")
    if target_mode not in {"auto", "external", "near_home"}:
        raise ValueError("target_mode must be one of: auto|external|near_home")

    from .sac_torch import SACTorchAgent, SACTorchConfig

    agent: Any = SACTorchAgent(
        SACTorchConfig(
            obs_dim=_OBS_DIM,
            action_dim=_CONTROLLED_ACTION_DIM,
            target_entropy=sac_target_entropy,
            action_scale=float(action_scale),
            mu_limit=float(actor_mu_limit),
            actor_update_delay=max(1, int(actor_update_delay)),
            bc_lambda=float(actor_bc_lambda),
            bc_outer_dpos_m=float(actor_bc_outer_dpos_m),
            bc_inner_dpos_m=float(actor_bc_inner_dpos_m),
            bc_topk=max(0, int(actor_bc_topk)),
            distill_lambda=float(actor_distill_lambda),
            distill_interval=max(1, int(actor_distill_interval)),
            distill_steps=max(1, int(actor_distill_steps)),
            distill_batch_size=max(0, int(actor_distill_batch_size)),
            distill_candidate_multiplier=max(1, int(actor_distill_candidate_multiplier)),
            distill_min_good_count=max(1, int(actor_distill_min_good_count)),
            distill_outer_dpos_m=float(actor_distill_outer_dpos_m),
            distill_support_dpos_m=float(actor_distill_support_dpos_m),
            distill_inner_dpos_m=float(actor_distill_inner_dpos_m),
            distill_dwell_dpos_m=float(actor_distill_dwell_dpos_m),
            distill_min_progress_m=float(actor_distill_min_progress_m),
            distill_max_delta_norm=float(actor_distill_max_delta_norm),
            distill_quality_threshold=float(actor_distill_quality_threshold),
            distill_advantage_beta=float(actor_distill_advantage_beta),
            distill_advantage_clip=float(actor_distill_advantage_clip),
            distill_grad_clip=float(actor_distill_grad_clip),
            distill_exclude_rejected=bool(actor_distill_exclude_rejected),
            distill_exclude_clamped=bool(actor_distill_exclude_clamped),
            distill_exclude_projected=bool(actor_distill_exclude_projected),
        ),
        seed=sac_seed,
    )

    loaded_from_checkpoint = False
    loaded_checkpoint_path: str | None = None
    for checkpoint_candidate in _checkpoint_candidates(artifact_root):
        if checkpoint_candidate.exists():
            _load_agent_checkpoint(agent, checkpoint_candidate)
            loaded_from_checkpoint = True
            loaded_checkpoint_path = str(checkpoint_candidate)
            break

    runtime = None
    runtime_controlled_indices: list[int] | None = None
    if runtime_mode == "gz":
        if not runtime_joint_names:
            raise ValueError("runtime_joint_names is required for runtime_mode=gz")
        runtime_controlled_indices = _controlled_joint_indices(runtime_joint_names)
        factory = runtime_factory or RuntimeROS2Adapter.from_ros2
        runtime_kwargs = {
            "joint_names": runtime_joint_names,
            "trajectory_topic": trajectory_topic,
            "joint_state_topic": joint_state_topic,
        }
        if runtime_factory is None:
            runtime_kwargs.update(
                {
                    "gz_visualize_target": bool(gz_visualize_target),
                    "gz_world_name": str(gz_world_name),
                    "gz_target_entity_name": str(gz_target_entity_name),
                    "gz_target_world_offset_xyz": (
                        float(gz_target_world_offset_x),
                        float(gz_target_world_offset_y),
                        float(gz_target_world_offset_z),
                    ),
                }
            )
        runtime = factory(**runtime_kwargs)

    external_ee_target, external_ee_target_source = _resolve_ee_target_from_external_task(
        prop=external_task_prop,
        src_idx=external_task_src_idx,
        dst_idx=external_task_dst_idx,
        waypoint_index=external_task_waypoint_index,
    )

    episodes_requested = max(1, int(episodes))
    successes = 0
    interventions = 0
    current_exploration_std_scale = float(exploration_std_scale)
    exploration_schedule_history: list[dict[str, Any]] = []
    global_best_min_dpos = float("inf")
    episode_outputs: list[dict[str, Any]] = []
    success_series: list[float] = []
    intervention_series: list[float] = []
    expected_log_lines_per_layer = 0
    reset_failures = 0
    reward_totals: list[float] = []
    train_metrics: list[dict[str, float]] = []
    episode_joint_delta_summary: list[dict[str, Any]] = []
    reset_failure_reasons: list[str] = []
    post_train_eval_summary: dict[str, Any] | None = None
    post_train_eval_artifacts: dict[str, str] = {}
    gap_eval_summary: dict[str, Any] | None = None
    gap_eval_artifacts: dict[str, str] = {}
    periodic_eval_history: list[dict[str, Any]] = []
    entropy_checkpoint_paths: list[str] = []
    best_eval_score = float("-inf")
    best_checkpoint_score = float("-inf")
    best_eval_episode = -1
    best_checkpoint_episode = -1
    best_checkpoint_path: str | None = None
    best_resume_count = 0
    last_best_resume_episode = -1
    early_stopped = False
    early_stop_reason: str | None = None
    target_curriculum = TargetCurriculumManager(
        TargetCurriculumStage(
            name="TC2",
            pos_offset_min_m=float(near_home_pos_offset_min_m),
            pos_offset_max_m=float(near_home_pos_offset_max_m),
            ori_offset_min_deg=float(near_home_ori_offset_min_deg),
            ori_offset_max_deg=float(near_home_ori_offset_max_deg),
        ),
        max_stage_index=target_curriculum_max_stage,
    )
    eval_suite_size = max(1, int(max(periodic_eval_episodes, post_train_eval_episodes)))
    gap_eval_specs = _parse_gap_eval_scales(gap_eval_scales)
    if gap_eval_specs:
        eval_suite_size = max(eval_suite_size, int(gap_eval_episodes) if int(gap_eval_episodes) > 0 else int(post_train_eval_episodes))
    fixed_eval_suite = _build_fixed_eval_suite(
        suite_size=eval_suite_size,
        suite_seed=int(sac_seed) + 700_001,
        target_mode=target_mode,
        action_stage_name=curriculum.current_stage.name,
        target_curriculum_stage_name=target_curriculum.current_stage.name,
        near_home_profile=target_curriculum.current_stage.name,
        near_home_pos_offset_min_m=float(target_curriculum.current_stage.pos_offset_min_m),
        near_home_pos_offset_max_m=float(target_curriculum.current_stage.pos_offset_max_m),
        near_home_ori_offset_min_deg=float(target_curriculum.current_stage.ori_offset_min_deg),
        near_home_ori_offset_max_deg=float(target_curriculum.current_stage.ori_offset_max_deg),
        external_ee_target=external_ee_target,
        external_ee_target_source=external_ee_target_source,
    )
    best_checkpoint_metadata: dict[str, Any] | None = None
    entropy_ratios = _parse_float_list(entropy_anneal_ratios, [1.0, 0.70, 0.50])
    entropy_stage_names = _parse_stage_names(entropy_anneal_stage_names, len(entropy_ratios))
    entropy_annealer = EntropyAnnealManager(
        mode=str(entropy_anneal_mode),
        baseline_target_entropy=float(getattr(agent, "target_entropy", -float(_CONTROLLED_ACTION_DIM))),
        ratios=entropy_ratios,
        stage_names=entropy_stage_names,
        fixed_episode_thresholds=_parse_int_list(entropy_anneal_fixed_episodes, [20, 40]),
        min_episode=int(entropy_anneal_min_episode),
        window=int(entropy_anneal_window),
        max_stage_index=entropy_anneal_max_stage_index,
    )
    entropy_annealer.apply_to_agent(agent)
    active_distill_lambda = (
        float(actor_distill_lambda)
        if (
            not entropy_annealer.enabled
            or int(entropy_annealer.state.stage_index) >= max(0, int(distill_start_entropy_stage_index))
        )
        else 0.0
    )
    if hasattr(agent, "set_distill_mode"):
        agent.set_distill_mode(lambda_value=active_distill_lambda, stage_name=entropy_annealer.current_stage.name)
    if entropy_annealer.enabled:
        entropy_checkpoint_paths.append(
            _save_agent_checkpoint(
                agent,
                artifact_root / "train" / f"checkpoint_entropy_stage_{entropy_annealer.current_stage.name}_start.pt",
                run_id,
                metadata={
                    "checkpoint_kind": "entropy_stage_start",
                    "episode": -1,
                    "entropy_stage": entropy_annealer.current_stage.name,
                    "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                    "alpha": float(agent.alpha),
                    "active_distill_lambda": float(active_distill_lambda),
                    "action_stage_name": str(curriculum.current_stage.name),
                    "target_curriculum_stage_name": str(target_curriculum.current_stage.name),
                    "eval_suite": dict(fixed_eval_suite),
                    "entropy_annealing": entropy_annealer.to_artifact(),
                },
            )
        )

    _progress_log(
        "run_start "
        f"run_id={run_id} episodes={episodes_requested} steps_per_episode={steps_per_episode} "
        f"runtime_mode={runtime_mode} policy_mode={policy_mode} stage_profile={stage_profile} "
        f"target_mode={target_mode} artifact_root={artifact_root} "
        f"exploration_std_scale={float(exploration_std_scale):.3f} "
        f"action_scale={float(action_scale):.3f} "
        f"sac_target_entropy={float(getattr(agent, 'target_entropy', 0.0)):.3f} "
        f"reward_profile={str(reward_profile)} "
        f"actor_mu_limit={float(actor_mu_limit):.3f} "
        f"actor_update_delay={int(actor_update_delay)} "
        f"actor_bc_lambda={float(actor_bc_lambda):.3f} "
        f"actor_distill_lambda={float(actor_distill_lambda):.3f} "
        f"actor_distill_interval={int(actor_distill_interval)} "
        f"periodic_eval_interval={int(periodic_eval_interval)} "
        f"entropy_anneal_mode={str(entropy_anneal_mode)} "
        f"entropy_stage={entropy_annealer.current_stage.name} "
        f"target_entropy={float(entropy_annealer.current_stage.target_entropy):.3f} "
        f"distill_start_entropy_stage_index={int(distill_start_entropy_stage_index)} "
        f"active_distill_lambda={float(active_distill_lambda):.3f} "
        f"disable_exploration_schedule={bool(disable_exploration_schedule)} "
        f"eval_suite_id={fixed_eval_suite['suite_id']}"
    )
    if loaded_from_checkpoint and loaded_checkpoint_path is not None:
        _progress_log(f"resume checkpoint={loaded_checkpoint_path}")

    try:
        for ep in range(episodes_requested):
            stage = curriculum.current_stage
            if int(stage.controlled_dofs) != _CONTROLLED_ACTION_DIM:
                raise ValueError(f"unsupported controlled_dofs={stage.controlled_dofs}, expected {_CONTROLLED_ACTION_DIM}")
            ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
            step_count = min(int(steps_per_episode), stage.step_budget)

            resolved_target_mode = target_mode
            if resolved_target_mode == "auto":
                resolved_target_mode = "near_home" if stage.name == "S0_B" else "external"

            _progress_log(
                f"episode_start ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                f"stage={stage.name} step_budget={step_count} target_mode={resolved_target_mode}"
            )

            home_q_for_target = _HOME_Q.copy()
            if runtime_mode == "gz" and runtime is not None and runtime_controlled_indices is not None:
                q_home_full = runtime.read_q()
                home_q_for_target = np.asarray(q_home_full, dtype=float)[np.asarray(runtime_controlled_indices, dtype=int)]

            if resolved_target_mode == "near_home":
                target_stage = target_curriculum.current_stage
                ee_target, ee_target_source = _resolve_near_home_ee_target(
                    home_q=home_q_for_target,
                    profile=target_stage.name,
                    pos_offset_min_m=target_stage.pos_offset_min_m,
                    pos_offset_max_m=target_stage.pos_offset_max_m,
                    ori_offset_min_deg=target_stage.ori_offset_min_deg,
                    ori_offset_max_deg=target_stage.ori_offset_max_deg,
                    rng=target_rng,
                )
            else:
                ee_target = np.asarray(external_ee_target, dtype=float).copy()
                ee_target_source = dict(external_ee_target_source)

            def _policy_fn(obs: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
                action, diagnostics = agent.act_with_diagnostics(
                    obs,
                    stochastic=True,
                    exploration_std_scale=float(current_exploration_std_scale),
                )
                return action, policy_mode, diagnostics

            reset_info = {"status": "skipped", "accepted": True, "execution_ok": True, "result_status": "skipped"}
            target_visualization = {
                "success": False,
                "action": "disabled",
                "reason": "not_attempted",
                "world_name": gz_world_name if runtime_mode == "gz" else None,
                "entity_name": gz_target_entity_name if runtime_mode == "gz" else None,
                "pose_offset_xyz": (
                    [float(gz_target_world_offset_x), float(gz_target_world_offset_y), float(gz_target_world_offset_z)]
                    if runtime_mode == "gz"
                    else None
                ),
            }
            try:
                if runtime_mode == "smoke":
                    def _smoke_policy_fn(q: np.ndarray, target_q: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
                        ee_pose = _ee_pose_from_q(q)
                        ee_pos_err, ee_ori_err = _ee_errors(ee_pose, ee_target)
                        obs = _obs_from_state(q=q, dq=np.zeros_like(q), ee_pose_err=np.concatenate([ee_pos_err, ee_ori_err]), prev_action=np.zeros_like(q))
                        action, policy_name, _diagnostics = _policy_fn(obs)
                        return action, policy_name, _diagnostics

                    logs = run_smoke(
                        run_id=ep_id,
                        steps=step_count,
                        log_root=logs_root,
                        episode=ep,
                        policy_fn=_smoke_policy_fn,
                        action_limit=float(stage.action_limit),
                        target_q=_HOME_Q,
                    )
                else:
                    reset_info = _reset_episode_home(
                        runtime=runtime,
                        controlled_indices=runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)),
                        home_q=_HOME_Q,
                        reset_near_home_eps=reset_near_home_eps,
                    )
                    _jsonl_append(runtime_trace_path, {"run_id": run_id, "episode": ep, "step": -1, "event": "reset", **reset_info})
                    if not (reset_info["accepted"] and reset_info["execution_ok"]):
                        _progress_log(
                            f"reset_fail ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                            f"reason={reset_info.get('fail_reason', 'unknown')}"
                        )
                        reset_failures += 1
                        reset_failure_reasons.append(
                            f"ep={ep}:reset:{reset_info.get('fail_reason','unknown')}"
                        )
                        _jsonl_append(
                            episode_reward_summary_path,
                            {
                                "run_id": run_id,
                                "episode": ep,
                                "episode_id": ep_id,
                                "reset_result": reset_info,
                                "reset_fail": True,
                                "reset_skipped_near_home": bool(reset_info.get("reset_skipped_near_home", False)),
                            },
                        )
                        break

                    if resolved_target_mode == "near_home":
                        target_stage = target_curriculum.current_stage
                        q_after_reset = np.asarray(reset_info.get("q_after", home_q_for_target.tolist()), dtype=float)
                        home_q_after_reset = q_after_reset[np.asarray(runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)), dtype=int)]
                        ee_target, ee_target_source = _resolve_near_home_ee_target(
                            home_q=home_q_after_reset,
                            profile=target_stage.name,
                            pos_offset_min_m=target_stage.pos_offset_min_m,
                            pos_offset_max_m=target_stage.pos_offset_max_m,
                            ori_offset_min_deg=target_stage.ori_offset_min_deg,
                            ori_offset_max_deg=target_stage.ori_offset_max_deg,
                            rng=target_rng,
                        )

                    publish_target_visual = getattr(runtime, "publish_ee_target_visual", None)
                    if callable(publish_target_visual):
                        target_visualization = dict(publish_target_visual(ee_target))
                        if bool(target_visualization.get("success", False)):
                            _progress_log(
                                f"target_visual ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                                f"action={target_visualization.get('action', 'unknown')} "
                                f"world={target_visualization.get('world_name', gz_world_name)} "
                                f"entity={target_visualization.get('entity_name', gz_target_entity_name)}"
                            )
                        else:
                            _progress_log(
                                f"target_visual_warn ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                                f"reason={target_visualization.get('reason', 'unknown')} "
                                f"world={target_visualization.get('world_name', gz_world_name)} "
                                f"entity={target_visualization.get('entity_name', gz_target_entity_name)}"
                            )

                    last_err: Exception | None = None
                    for _attempt in range(2):
                        try:
                            logs = _run_episode_gz(
                                ep_id=ep_id,
                                ep_index=ep,
                                step_count=step_count,
                                logs_root=logs_root,
                                runtime=runtime,
                                ee_target=ee_target,
                                controlled_indices=runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)),
                                policy_fn=_policy_fn,
                                action_limit=float(stage.action_limit),
                                reward_config=reward_composer.config,
                                success_dwell_pos_m=float(reward_composer.config.dwell_pos_m),
                                success_dwell_steps_required=int(reward_composer.config.dwell_steps_required),
                            )
                            last_err = None
                            break
                        except Exception as e:  # transient startup/readback races
                            last_err = e
                            time.sleep(0.25)
                    if last_err is not None:
                        raise last_err
            except Exception as e:
                _progress_log(
                    f"episode_abort ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                    f"error={type(e).__name__}:{e}"
                )
                reset_failures += 1
                reset_failure_reasons.append(f"ep={ep}:{type(e).__name__}:{e}")
                break

            expected_log_lines_per_layer += max(1, int(step_count))

            episode_return = 0.0
            ep_intervention = 0
            prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
            trace_steps = logs.get("trace_steps", [])
            reward_composer.reset_episode_state()
            ep_component_sums: dict[str, float] = {
                "progress": 0.0,
                "action": 0.0,
                "jerk": 0.0,
                "adjust_penalty": 0.0,
                "raw_action_penalty": 0.0,
                "reject_penalty": 0.0,
                "intervention": 0.0,
                "clamp_or_projection": 0.0,
                "stall": 0.0,
                "ee_small_motion_penalty": 0.0,
                "timeout_penalty": 0.0,
                "reset_fail_penalty": 0.0,
                "execution_fail_penalty": 0.0,
                "timeout_or_reset": 0.0,
                "success_bonus": 0.0,
            "near_goal": 0.0,
            "near_goal_shell": 0.0,
            "inner_shell": 0.0,
            "dwell": 0.0,
            "outer_exit": 0.0,
            "inner_exit": 0.0,
            "zone_exit": 0.0,
            "near_goal_exit": 0.0,
            "local_drift_penalty": 0.0,
            "dwell_break": 0.0,
                "ori_progress": 0.0,
                "reward_total": 0.0,
            }
            ep_max_dwell_count = 0
            ep_dwell_break_count = 0
            ep_clamp_count = 0
            ep_projection_count = 0
            ep_reject_count = 0
            ep_sum_delta_norm = 0.0
            ep_sum_adjust_penalty = 0.0
            ep_sum_raw_action_penalty = 0.0
            ep_sum_reject_penalty = 0.0
            ep_near_goal_entry_count = 0
            ep_near_goal_shell_count = 0
            ep_inner_shell_count = 0
            ep_near_goal_exit_count = 0
            ep_zone_exit_count = 0
            ep_success_triggered_by_dwell = False
            ep_min_dpos = float("inf")
            ep_final_dpos = 0.0
            ep_drift_sum = 0.0

            ep_q_before = np.asarray(trace_steps[0]["runtime"]["q_before"], dtype=float) if (trace_steps and runtime_mode == "gz") else None
            ep_q_after = np.asarray(trace_steps[-1]["runtime"]["q_after"], dtype=float) if (trace_steps and runtime_mode == "gz") else None
            terminal_done = False
            terminal_done_reason = "none"
            terminal_step: int | None = None

            for idx, step in enumerate(trace_steps):
                action_raw = np.asarray(step["action_raw"], dtype=float)
                action_exec = np.asarray(step.get("action_exec", step.get("action_clamped", step["action_raw"])), dtype=float)
                delta_a = np.asarray(step.get("delta_a", action_exec - action_raw), dtype=float)
                delta_norm = float(step.get("delta_norm", _scaled_action_norm(delta_a, reward_composer.config.action_scale)))
                raw_norm = float(step.get("raw_norm", _scaled_action_norm(action_raw, reward_composer.config.action_scale)))
                exec_norm = float(step.get("exec_norm", _scaled_action_norm(action_exec, reward_composer.config.action_scale)))
                step_ee_target = np.asarray(step.get("ee_target", ee_target.tolist()), dtype=float)
                step_ee_pose = np.asarray(step.get("ee_pose", np.zeros(6, dtype=float)), dtype=float)
                prev_ee_pos_err = np.asarray(step_ee_target[:3], dtype=float) - np.asarray(step_ee_pose[:3], dtype=float)
                prev_ee_ori_err = np.asarray(step_ee_target[3:6], dtype=float) - np.asarray(step_ee_pose[3:6], dtype=float)
                curr_ee_pos_err = np.asarray(step.get("ee_pos_err", prev_ee_pos_err), dtype=float)
                curr_ee_ori_err = np.asarray(step.get("ee_ori_err", prev_ee_ori_err), dtype=float)
                prev_error = float(step.get("goal_error_prev", RewardComposer.ee_error_norm(prev_ee_pos_err, prev_ee_ori_err)))
                curr_error = float(step.get("goal_error_next", RewardComposer.ee_error_norm(curr_ee_pos_err, curr_ee_ori_err)))
                ee_step_dpos = float(np.linalg.norm(curr_ee_pos_err - prev_ee_pos_err))
                ee_step_dori = float(np.linalg.norm(curr_ee_ori_err - prev_ee_ori_err))
                intervention_now = step["intervention"] != "none"
                clamp_or_projection = bool(step["saturated"] or step["projection_applied"])
                execution_ok = bool(step.get("runtime", {}).get("execution_ok", True))

                done = idx == len(trace_steps) - 1
                done_reason = "running"
                if not execution_ok:
                    done = True
                    done_reason = "execution_fail"
                elif done:
                    if bool(step.get("success_by_dwell", False)):
                        done_reason = "success"
                    elif step["intervention"] == "no_effect":
                        done_reason = "no_effect"
                    else:
                        done_reason = "success" if (float(np.linalg.norm(curr_ee_pos_err)) < float(ee_pos_success_threshold) and float(np.linalg.norm(curr_ee_ori_err)) < float(ee_ori_success_threshold)) else "timeout"

                if done:
                    terminal_done = True
                    terminal_done_reason = done_reason
                    terminal_step = int(step["step"])

                q_before_arr = np.asarray(step.get("obs_q", []), dtype=float)
                q_after_arr = np.asarray(step.get("q_after", []), dtype=float)
                q_before_for_stall = q_before_arr if q_before_arr.size > 0 and q_before_arr.shape == q_after_arr.shape else None
                q_after_for_stall = q_after_arr if q_before_for_stall is not None else None

                reward_state_in = reward_composer.state_dict()
                prev_action_in = prev_action.copy()
                terms = reward_composer.compute(
                    prev_ee_pos_err=prev_ee_pos_err,
                    prev_ee_ori_err=prev_ee_ori_err,
                    curr_ee_pos_err=curr_ee_pos_err,
                    curr_ee_ori_err=curr_ee_ori_err,
                    action=action_exec,
                    action_raw=action_raw,
                    action_exec=action_exec,
                    prev_action=prev_action_in,
                    intervention=intervention_now,
                    clamp_or_projection=clamp_or_projection,
                    rejected=bool(step.get("rejected", False)),
                    done=done,
                    done_reason=done_reason,
                    q_before=q_before_for_stall,
                    q_after=q_after_for_stall,
                    effect_ratio=(step.get("runtime", {}) or {}).get("effect_ratio"),
                )
                reward_state_out = reward_composer.state_dict()
                prev_action = action_exec
                terms_dict = terms.to_dict()
                episode_return += terms.reward_total
                reward_totals.append(terms.reward_total)
                for k in ep_component_sums:
                    ep_component_sums[k] += float(terms_dict[k])
                ep_sum_adjust_penalty += float(terms.adjust_penalty)
                ep_sum_raw_action_penalty += float(terms.raw_action_penalty)
                ep_sum_reject_penalty += float(terms.reject_penalty)
                ep_max_dwell_count = max(ep_max_dwell_count, int(terms.dwell_count))
                ep_dwell_break_count += int(terms.dwell_break != 0.0)
                ep_clamp_count += int(clamp_or_projection)
                ep_projection_count += int(bool(step.get("projection_applied", False)))
                ep_reject_count += int(bool(step.get("rejected", False)))
                ep_sum_delta_norm += delta_norm
                ep_near_goal_entry_count += int(terms.near_goal != 0.0)
                ep_near_goal_shell_count += int(terms.in_near_goal_shell > 0.5)
                ep_inner_shell_count += int(terms.in_inner_shell > 0.5)
                ep_near_goal_exit_count += int(terms.near_goal_exit != 0.0)
                ep_zone_exit_count += int(terms.zone_exit != 0.0)
                ep_success_triggered_by_dwell = ep_success_triggered_by_dwell or bool(terms.success_triggered_by_dwell)
                ep_drift_sum += float(terms.local_drift_penalty)
                curr_dpos = float(np.linalg.norm(curr_ee_pos_err))
                ep_min_dpos = min(ep_min_dpos, curr_dpos)
                ep_final_dpos = curr_dpos

                should_log_step = (
                    idx == 0
                    or done
                    or terms.success_triggered_by_dwell > 0.0
                    or ((idx + 1) % _PROGRESS_LOG_EVERY_STEPS == 0)
                )
                if should_log_step:
                    _progress_log(
                        f"episode_step ep={ep + 1}/{episodes_requested} step={idx + 1}/{len(trace_steps)} "
                        f"dpos={curr_dpos:.4f} reward={terms.reward_total:+.4f} "
                        f"dwell={int(terms.dwell_count)} clamp={int(clamp_or_projection)} "
                        f"near_entry={int(terms.near_goal != 0.0)} "
                        f"delta_norm={delta_norm:.3f} reject={int(bool(step.get('rejected', False)))} "
                        f"sat_dims={int((step.get('policy_debug') or {}).get('saturated_dims', 0))} "
                        f"pre_tanh_max={float((step.get('policy_debug') or {}).get('pre_tanh_abs_max', 0.0)):.3f} "
                        f"done={done_reason}"
                    )

                reward_trace.append(
                    {
                        "run_id": run_id,
                        "episode": ep,
                        "episode_id": ep_id,
                        "step": int(step["step"]),
                        "policy_mode": policy_mode,
                        "runtime_mode": runtime_mode,
                        "done": done,
                        "done_reason": done_reason,
                        "goal_error_prev": prev_error,
                        "goal_error_next": curr_error,
                        "ee_pose": step.get("ee_pose"),
                        "ee_target": step.get("ee_target"),
                        "ee_pos_err": step.get("ee_pos_err"),
                        "ee_ori_err": step.get("ee_ori_err"),
                        "action_raw": action_raw.tolist(),
                        "action_exec": action_exec.tolist(),
                        "delta_a": delta_a.tolist(),
                        "raw_norm": raw_norm,
                        "exec_norm": exec_norm,
                        "delta_norm": delta_norm,
                        "clamp_triggered": bool(step.get("clamp_triggered", step.get("saturated", False))),
                        "projection_triggered": bool(step.get("projection_applied", False)),
                        "rejected": bool(step.get("rejected", False)),
                        "reject_reason": step.get("reject_reason", "none"),
                        "ee_step_dpos": ee_step_dpos,
                        "ee_step_dori": ee_step_dori,
                        "ee_small_motion_penalty": float(terms.ee_small_motion_penalty),
                        "reward_total": float(terms.reward_total),
                        "policy_debug": step.get("policy_debug", {}),
                        "reward_state_in": reward_state_in,
                        "reward_state_out": reward_state_out,
                        "components": terms_dict,
                    }
                )

                if runtime_mode == "gz":
                    _jsonl_append(
                        runtime_trace_path,
                        {
                            "run_id": run_id,
                            "episode": ep,
                            "step": int(step["step"]),
                            "cmd_q": step["runtime"]["cmd_q"],
                            "readback_q_before": step["runtime"]["q_before"],
                            "readback_q_after": step["runtime"]["q_after"],
                            "joint_delta": step["runtime"]["joint_delta"],
                            "joint_delta_l2": step["runtime"]["joint_delta_l2"],
                            "cmd_delta_l2": step["runtime"].get("cmd_delta_l2"),
                            "effect_ratio": step["runtime"].get("effect_ratio"),
                            "frame_before_stamp_ns": step["runtime"].get("frame_before_stamp_ns"),
                            "frame_after_stamp_ns": step["runtime"].get("frame_after_stamp_ns"),
                            "goal_error_prev": prev_error,
                            "goal_error_next": curr_error,
                            "ee_pose": step.get("ee_pose"),
                            "ee_target": step.get("ee_target"),
                            "ee_pos_err": step.get("ee_pos_err"),
                            "ee_ori_err": step.get("ee_ori_err"),
                            "action_raw": action_raw.tolist(),
                            "action_exec": action_exec.tolist(),
                            "delta_a": delta_a.tolist(),
                            "raw_norm": raw_norm,
                            "exec_norm": exec_norm,
                            "delta_norm": delta_norm,
                            "clamp_triggered": bool(step.get("clamp_triggered", step.get("saturated", False))),
                            "projection_triggered": bool(step.get("projection_applied", False)),
                            "rejected": bool(step.get("rejected", False)),
                            "reject_reason": step.get("reject_reason", "none"),
                            "policy_debug": step.get("policy_debug", {}),
                            "no_effect": bool(step.get("no_effect", False)),
                            "no_effect_reason": step["runtime"].get("no_effect_reason", "none"),
                            "no_effect_streak": int(step.get("no_effect_streak", 0)),
                            "skipped_publish": bool(step["runtime"].get("skipped_publish", False)),
                            "accepted": bool(step["runtime"].get("accepted", True)),
                            "result_status": step["runtime"].get("result_status", "success"),
                            "execution_ok": bool(step["runtime"].get("execution_ok", True)),
                            "fail_reason": step["runtime"].get("fail_reason", "none"),
                            "command_path": step["runtime"].get("command_path", "topic_fallback"),
                            "action_status": step["runtime"].get("action_status"),
                            "action_error_code": step["runtime"].get("action_error_code"),
                            "timestamp_ns": step["runtime"]["timestamp_ns"],
                        },
                    )

                if intervention_now:
                    ep_intervention = 1

                q_now = np.asarray(step["obs_q"], dtype=float)
                q_next = np.asarray(step["q_after"], dtype=float) if runtime_mode == "gz" else (q_now + np.asarray(step["action_clamped"], dtype=float))
                dq_now = np.zeros_like(q_now) if idx == 0 else (q_now - np.asarray(trace_steps[idx - 1]["obs_q"], dtype=float))
                dq_next = q_next - q_now
                ee_pose_err_now = np.concatenate([prev_ee_pos_err, prev_ee_ori_err])
                ee_pose_err_next = np.concatenate([curr_ee_pos_err, curr_ee_ori_err])
                obs = _obs_from_state(q=q_now, dq=dq_now, ee_pose_err=ee_pose_err_now, prev_action=prev_action_in)
                next_obs = _obs_from_state(q=q_next, dq=dq_next, ee_pose_err=ee_pose_err_next, prev_action=action_exec)
                agent.remember(
                    obs,
                    action_raw,
                    action_exec,
                    terms.reward_total,
                    next_obs,
                    done,
                    info={
                        "prev_q_des": np.asarray(step.get("prev_q_des_in", q_now), dtype=float),
                        "next_prev_q_des": np.asarray(step.get("q_des_applied", q_next), dtype=float),
                        "delta_limits": np.asarray(step.get("delta_limits", np.full(_CONTROLLED_ACTION_DIM, float(stage.action_limit))), dtype=float),
                        "delta_norm": float(delta_norm),
                        "raw_norm": float(raw_norm),
                        "exec_norm": float(exec_norm),
                        "clamp_triggered": bool(step.get("clamp_triggered", step.get("saturated", False))),
                        "projection_triggered": bool(step.get("projection_applied", False)),
                        "rejected": bool(step.get("rejected", False)),
                        "success": bool(done_reason == "success" or terms.success_triggered_by_dwell),
                        "dwell_count": int(terms.dwell_count),
                    },
                )
                train_out = agent.train_step()
                if train_out is not None:
                    train_metrics.append(train_out)
                    _jsonl_append(
                        train_metrics_path,
                        {
                            "run_id": run_id,
                            "episode": int(ep),
                            "step": int(step["step"]),
                            "entropy_stage": str(entropy_annealer.current_stage.name),
                            "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                            "active_distill_lambda": float(getattr(agent, "active_distill_lambda", 0.0)),
                            **train_out,
                        },
                    )

                if done_reason == "execution_fail":
                    ep_intervention = 1
                    break

            final_error = float(logs.get("final_goal_error", 1.0))
            if trace_steps:
                last = trace_steps[-1]
                pos_ok = float(np.linalg.norm(np.asarray(last.get("ee_pos_err", [9, 9, 9]), dtype=float))) < float(ee_pos_success_threshold)
                ori_ok = float(np.linalg.norm(np.asarray(last.get("ee_ori_err", [9, 9, 9]), dtype=float))) < float(ee_ori_success_threshold)
                ep_success = 1 if (bool(last.get("success_by_dwell", False)) or (pos_ok and ori_ok)) else 0
            else:
                ep_success = 0
            zone_metrics = _dpos_zone_metrics(
                min_dpos=float(ep_min_dpos if trace_steps else 0.0),
                final_dpos=float(ep_final_dpos if trace_steps else 0.0),
                reward_config=reward_composer.config,
            )

            if runtime_mode == "gz" and ep_q_before is not None and ep_q_after is not None:
                delta = ep_q_after - ep_q_before
                episode_joint_delta_summary.append(
                    {
                        "episode": ep,
                        "before_q": ep_q_before.tolist(),
                        "after_q": ep_q_after.tolist(),
                        "delta_q": delta.tolist(),
                        "delta_q_l2": float(np.linalg.norm(delta)),
                    }
                )

            record = curriculum.record_episode(success_rate=float(ep_success))
            successes += ep_success
            interventions += ep_intervention
            if trace_steps:
                global_best_min_dpos = min(float(global_best_min_dpos), float(ep_min_dpos))
            success_series.append(float(ep_success))
            intervention_series.append(float(ep_intervention))
            updates_applied_now = int(train_metrics[-1].get("updates_applied", 0.0)) if train_metrics else 0
            _progress_log(
                f"episode_end ep={ep + 1}/{episodes_requested} ep_id={ep_id} "
                f"success={ep_success} done_reason={terminal_done_reason} "
                f"return={episode_return:+.4f} min_dpos={ep_min_dpos if trace_steps else 0.0:.4f} "
                f"final_dpos={ep_final_dpos if trace_steps else 0.0:.4f} "
                f"max_dwell={ep_max_dwell_count} dwell_breaks={ep_dwell_break_count} "
                f"clamp_count={ep_clamp_count} reject_count={ep_reject_count} "
                f"near_entries={ep_near_goal_entry_count} "
                f"updates_applied={updates_applied_now}"
            )

            step_n = max(1, len(trace_steps))
            ep_summary = {
                "run_id": run_id,
                "episode": ep,
                "episode_id": ep_id,
                "stage": record.stage_name,
                "component_sums": ep_component_sums,
                "component_means": {k: float(v) / float(step_n) for k, v in ep_component_sums.items()},
                "total_reward": float(ep_component_sums["reward_total"]),
                "success_count": int(ep_success),
                "intervention_count": int(ep_intervention),
                "no_effect_count": int(1 if trace_steps and trace_steps[-1]["intervention"] == "no_effect" else 0),
                "target_mode": resolved_target_mode,
                "target_curriculum_stage": target_curriculum.current_stage.name,
                "home_ee": ee_target_source.get("home_ee"),
                "ee_target": ee_target.tolist(),
                "target_delta_pos": ee_target_source.get("target_delta_pos"),
                "target_delta_ori": ee_target_source.get("target_delta_ori"),
                "target_visualization": target_visualization,
                "max_dwell_count": int(ep_max_dwell_count),
                "dwell_break_count": int(ep_dwell_break_count),
                "clamp_count": int(ep_clamp_count),
                "projection_count": int(ep_projection_count),
                "reject_count": int(ep_reject_count),
                "reject_rate": _safe_rate(ep_reject_count, len(trace_steps)),
                "exploration_std_scale": float(current_exploration_std_scale),
                "entropy_stage": entropy_annealer.current_stage.name,
                "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                "alpha": float(agent.alpha),
                "near_goal_entry_count": int(ep_near_goal_entry_count),
                "near_goal_shell_count": int(ep_near_goal_shell_count),
                "inner_shell_count": int(ep_inner_shell_count),
                "near_goal_exit_count": int(ep_near_goal_exit_count),
                "zone_exit_count": int(ep_zone_exit_count),
                "shell_hit": bool(ep_near_goal_shell_count > 0),
                "inner_shell_hit": bool(ep_inner_shell_count > 0),
                "dwell_hit": bool(ep_max_dwell_count > 0),
                "drift_sum": float(ep_drift_sum),
                "success_triggered_by_dwell": bool(ep_success_triggered_by_dwell),
                "sum_adjust_penalty": float(ep_sum_adjust_penalty),
                "sum_raw_action_penalty": float(ep_sum_raw_action_penalty),
                "sum_reject_penalty": float(ep_sum_reject_penalty),
                "sum_delta_norm": float(ep_sum_delta_norm),
                "min_dpos": float(ep_min_dpos if trace_steps else 0.0),
                "final_dpos": float(ep_final_dpos if trace_steps else 0.0),
                "final_dpos_minus_min_dpos": float((ep_final_dpos - ep_min_dpos) if trace_steps else 0.0),
                **zone_metrics,
                "reset_result": reset_info,
                "reset_skipped_near_home": bool(reset_info.get("reset_skipped_near_home", False)),
                "terminal": bool(terminal_done),
                "done_reason": terminal_done_reason,
                "terminal_step": terminal_step,
            }
            _jsonl_append(episode_reward_summary_path, ep_summary)

            episode_outputs.append(
                {
                    "episode": ep,
                    "run_id": ep_id,
                    "stage": record.stage_name,
                    "success_rate": float(ep_success),
                    "promoted": record.promoted,
                    "logs": {k: v for k, v in logs.items() if k in {"l1", "l2", "l3"}},
                    "has_intervention": bool(ep_intervention),
                    "episode_return": float(episode_return),
                    "final_goal_error": final_error,
                    "policy_mode": policy_mode,
                    "runtime_mode": runtime_mode,
                    "target_mode": resolved_target_mode,
                    "target_curriculum_stage": target_curriculum.current_stage.name,
                    "home_ee": ee_target_source.get("home_ee"),
                    "ee_target": ee_target.tolist(),
                    "target_delta_pos": ee_target_source.get("target_delta_pos"),
                    "target_delta_ori": ee_target_source.get("target_delta_ori"),
                    "target_visualization": target_visualization,
                    "max_dwell_count": int(ep_max_dwell_count),
                    "dwell_break_count": int(ep_dwell_break_count),
                    "clamp_count": int(ep_clamp_count),
                    "projection_count": int(ep_projection_count),
                    "reject_count": int(ep_reject_count),
                    "reject_rate": _safe_rate(ep_reject_count, len(trace_steps)),
                    "exploration_std_scale": float(current_exploration_std_scale),
                    "entropy_stage": entropy_annealer.current_stage.name,
                    "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                    "alpha": float(agent.alpha),
                    "near_goal_entry_count": int(ep_near_goal_entry_count),
                    "near_goal_shell_count": int(ep_near_goal_shell_count),
                    "inner_shell_count": int(ep_inner_shell_count),
                    "near_goal_exit_count": int(ep_near_goal_exit_count),
                    "zone_exit_count": int(ep_zone_exit_count),
                    "shell_hit": bool(ep_near_goal_shell_count > 0),
                    "inner_shell_hit": bool(ep_inner_shell_count > 0),
                    "dwell_hit": bool(ep_max_dwell_count > 0),
                    "drift_sum": float(ep_drift_sum),
                    "success_triggered_by_dwell": bool(ep_success_triggered_by_dwell),
                    "sum_adjust_penalty": float(ep_sum_adjust_penalty),
                    "sum_raw_action_penalty": float(ep_sum_raw_action_penalty),
                    "sum_reject_penalty": float(ep_sum_reject_penalty),
                    "sum_delta_norm": float(ep_sum_delta_norm),
                    "min_dpos": float(ep_min_dpos if trace_steps else 0.0),
                    "final_dpos": float(ep_final_dpos if trace_steps else 0.0),
                    "final_dpos_minus_min_dpos": float((ep_final_dpos - ep_min_dpos) if trace_steps else 0.0),
                    **zone_metrics,
                    "ee_target_source": ee_target_source,
                    "reset_result": reset_info,
                    "terminal": bool(terminal_done),
                    "done_reason": terminal_done_reason,
                    "terminal_step": terminal_step,
                }
            )

            should_run_periodic_eval = (
                runtime_mode == "gz"
                and runtime is not None
                and runtime_controlled_indices is not None
                and int(periodic_eval_interval) > 0
                and ((ep + 1) % int(periodic_eval_interval) == 0)
                and int(periodic_eval_episodes) > 0
            )
            if should_run_periodic_eval:
                periodic_eval_root = artifact_root / "eval_periodic" / f"ep{ep + 1:03d}"
                periodic_eval = _run_post_training_eval_gz(
                    run_id=f"{run_id}_periodic_ep{ep + 1:03d}",
                    artifact_root=artifact_root,
                    eval_root=periodic_eval_root,
                    agent=agent,
                    runtime=runtime,
                    stage=curriculum.current_stage,
                    target_mode=target_mode,
                    near_home_profile=str(fixed_eval_suite["near_home_profile"]),
                    near_home_pos_offset_min_m=float(fixed_eval_suite["near_home_pos_offset_min_m"]),
                    near_home_pos_offset_max_m=float(fixed_eval_suite["near_home_pos_offset_max_m"]),
                    near_home_ori_offset_min_deg=float(fixed_eval_suite["near_home_ori_offset_min_deg"]),
                    near_home_ori_offset_max_deg=float(fixed_eval_suite["near_home_ori_offset_max_deg"]),
                    external_ee_target=external_ee_target,
                    external_ee_target_source=external_ee_target_source,
                    runtime_controlled_indices=runtime_controlled_indices,
                    policy_mode=policy_mode,
                    runtime_mode=runtime_mode,
                    episodes=int(periodic_eval_episodes),
                    steps_per_episode=int(eval_step_budget),
                    ee_pos_success_threshold=ee_pos_success_threshold,
                    ee_ori_success_threshold=ee_ori_success_threshold,
                    reset_near_home_eps=reset_near_home_eps,
                    sac_seed=sac_seed,
                gz_world_name=gz_world_name,
                gz_target_entity_name=gz_target_entity_name,
                gz_target_world_offset_xyz=(
                    float(gz_target_world_offset_x),
                    float(gz_target_world_offset_y),
                    float(gz_target_world_offset_z),
                ),
                reward_config=reward_composer.config,
                fixed_eval_suite=fixed_eval_suite,
            )
                periodic_metrics = dict(periodic_eval["summary"].get("metrics", {}))
                # Periodic eval runs are deterministic, so expose their success rate explicitly.
                periodic_metrics["det_success_rate"] = float(periodic_metrics.get("success_rate", 0.0))
                det_action_stats = dict(periodic_eval["summary"].get("action_stats", {}))
                periodic_metrics["det_action_l2_mean"] = float(det_action_stats.get("final_action_l2_mean", 0.0))
                periodic_metrics["det_raw_norm_mean"] = float(det_action_stats.get("raw_norm_mean", 0.0))
                periodic_metrics["det_exec_norm_mean"] = float(det_action_stats.get("exec_norm_mean", 0.0))
                periodic_metrics["det_true_basin_hit_rate"] = float(periodic_metrics.get("true_basin_hit_rate", 0.0))
                periodic_metrics["det_true_outer_hit_rate"] = float(periodic_metrics.get("true_outer_hit_rate", 0.0))
                periodic_metrics["det_true_inner_hit_rate"] = float(periodic_metrics.get("true_inner_hit_rate", 0.0))
                periodic_metrics["det_true_dwell_hit_rate"] = float(periodic_metrics.get("true_dwell_hit_rate", 0.0))
                stochastic_periodic_eval: dict[str, Any] | None = None
                if entropy_annealer.enabled:
                    stoch_scale = float(current_exploration_std_scale)
                    stoch_label = f"stochastic_noise{int(round(stoch_scale * 100.0)):03d}"
                    stochastic_periodic_eval = _run_post_training_eval_gz(
                        run_id=f"{run_id}_periodic_ep{ep + 1:03d}_{stoch_label}",
                        artifact_root=artifact_root,
                        eval_root=periodic_eval_root / stoch_label,
                        agent=agent,
                        runtime=runtime,
                        stage=curriculum.current_stage,
                        target_mode=target_mode,
                        near_home_profile=str(fixed_eval_suite["near_home_profile"]),
                        near_home_pos_offset_min_m=float(fixed_eval_suite["near_home_pos_offset_min_m"]),
                        near_home_pos_offset_max_m=float(fixed_eval_suite["near_home_pos_offset_max_m"]),
                        near_home_ori_offset_min_deg=float(fixed_eval_suite["near_home_ori_offset_min_deg"]),
                        near_home_ori_offset_max_deg=float(fixed_eval_suite["near_home_ori_offset_max_deg"]),
                        external_ee_target=external_ee_target,
                        external_ee_target_source=external_ee_target_source,
                        runtime_controlled_indices=runtime_controlled_indices,
                        policy_mode=policy_mode,
                        runtime_mode=runtime_mode,
                        episodes=int(periodic_eval_episodes),
                        steps_per_episode=int(eval_step_budget),
                        ee_pos_success_threshold=ee_pos_success_threshold,
                        ee_ori_success_threshold=ee_ori_success_threshold,
                        reset_near_home_eps=reset_near_home_eps,
                        sac_seed=sac_seed,
                        gz_world_name=gz_world_name,
                        gz_target_entity_name=gz_target_entity_name,
                        gz_target_world_offset_xyz=(
                            float(gz_target_world_offset_x),
                            float(gz_target_world_offset_y),
                            float(gz_target_world_offset_z),
                        ),
                        reward_config=reward_composer.config,
                        fixed_eval_suite=fixed_eval_suite,
                        eval_name=stoch_label,
                        eval_stochastic=True,
                        eval_exploration_std_scale=stoch_scale,
                        artifact_key_prefix=f"periodic_{stoch_label}",
                    )
                    stoch_metrics = dict(stochastic_periodic_eval["summary"].get("metrics", {}))
                    stoch_action_stats = dict(stochastic_periodic_eval["summary"].get("action_stats", {}))
                    periodic_metrics["stoch_success_rate"] = float(stoch_metrics.get("success_rate", 0.0))
                    periodic_metrics["stoch_true_basin_hit_rate"] = float(stoch_metrics.get("true_basin_hit_rate", 0.0))
                    periodic_metrics["stoch_true_outer_hit_rate"] = float(stoch_metrics.get("true_outer_hit_rate", 0.0))
                    periodic_metrics["stoch_true_inner_hit_rate"] = float(stoch_metrics.get("true_inner_hit_rate", 0.0))
                    periodic_metrics["stoch_true_dwell_hit_rate"] = float(stoch_metrics.get("true_dwell_hit_rate", 0.0))
                    periodic_metrics["stoch_mean_final_dpos"] = float(stoch_metrics.get("mean_final_dpos", 0.0))
                    periodic_metrics["stoch_action_l2_mean"] = float(stoch_action_stats.get("final_action_l2_mean", 0.0))
                    periodic_metrics["stoch_raw_norm_mean"] = float(stoch_action_stats.get("raw_norm_mean", 0.0))
                    periodic_metrics["stoch_exec_norm_mean"] = float(stoch_action_stats.get("exec_norm_mean", 0.0))
                    periodic_metrics["det_action_l2_over_stoch_action_l2"] = (
                        float(periodic_metrics["det_action_l2_mean"])
                        / max(float(periodic_metrics["stoch_action_l2_mean"]), 1e-8)
                    )
                    periodic_metrics["det_raw_norm_over_stoch_raw_norm"] = (
                        float(periodic_metrics["det_raw_norm_mean"])
                        / max(float(periodic_metrics["stoch_raw_norm_mean"]), 1e-8)
                    )
                recent_train_window = episode_outputs[-max(1, int(periodic_eval_interval)) :]
                periodic_metrics["train_success_rate"] = float(
                    np.mean([float(ep_out.get("success_rate", 0.0)) for ep_out in recent_train_window])
                ) if recent_train_window else 0.0
                recent_entropy_window_metrics = _recent_episode_metrics(episode_outputs, entropy_annealer.window)
                recent_distill_metrics = _recent_train_update_metrics(
                    train_metrics,
                    max(1, int(entropy_annealer.window) * max(1, int(steps_per_episode))),
                )
                periodic_metrics.update({f"recent_{k}": float(v) for k, v in recent_distill_metrics.items()})
                periodic_score = _eval_score(periodic_metrics)
                periodic_checkpoint_score = _checkpoint_score(periodic_metrics)
                curriculum_event = target_curriculum.record_eval(ep, periodic_metrics, periodic_score)
                entropy_event = entropy_annealer.maybe_advance(
                    episode_index=ep,
                    agent=agent,
                    recent_train_metrics=recent_entropy_window_metrics,
                    eval_metrics=periodic_metrics,
                )
                active_distill_lambda = (
                    float(actor_distill_lambda)
                    if (
                        not entropy_annealer.enabled
                        or int(entropy_annealer.state.stage_index) >= max(0, int(distill_start_entropy_stage_index))
                    )
                    else 0.0
                )
                if hasattr(agent, "set_distill_mode"):
                    agent.set_distill_mode(
                        lambda_value=active_distill_lambda,
                        stage_name=entropy_annealer.current_stage.name,
                    )
                if entropy_event is not None:
                    entropy_checkpoint_path = _save_agent_checkpoint(
                        agent,
                        artifact_root
                        / "train"
                        / f"checkpoint_entropy_stage_{entropy_event['stage_after']}_start_ep{ep + 1:03d}.pt",
                        run_id,
                        metadata={
                            "checkpoint_kind": "entropy_stage_start",
                            "episode": int(ep),
                            "entropy_event": entropy_event,
                            "entropy_stage": entropy_event["stage_after"],
                            "target_entropy": float(entropy_event["target_entropy_after"]),
                            "alpha": float(agent.alpha),
                            "active_distill_lambda": float(active_distill_lambda),
                            "action_stage_name": str(curriculum.current_stage.name),
                            "target_curriculum_stage_name": str(target_curriculum.current_stage.name),
                            "eval_suite": dict(fixed_eval_suite),
                            "periodic_metrics": dict(periodic_metrics),
                            "recent_train_metrics": dict(recent_entropy_window_metrics),
                            "entropy_annealing": entropy_annealer.to_artifact(),
                        },
                    )
                    entropy_checkpoint_paths.append(entropy_checkpoint_path)
                    _progress_log(
                        f"entropy_anneal ep={ep + 1}/{episodes_requested} "
                        f"{entropy_event['stage_before']}->{entropy_event['stage_after']} "
                        f"target_entropy={float(entropy_event['target_entropy_after']):.3f} "
                        f"reason={entropy_event['reason']}"
                    )
                periodic_metrics["entropy_stage"] = str(entropy_annealer.current_stage.name)
                periodic_metrics["target_entropy"] = float(entropy_annealer.current_stage.target_entropy)
                periodic_metrics["alpha"] = float(agent.alpha)
                periodic_metrics["active_distill_lambda"] = float(active_distill_lambda)
                periodic_record = {
                    "episode": int(ep),
                    "eval_root": str(periodic_eval_root),
                    "score": float(periodic_checkpoint_score),
                    "score_curriculum": float(periodic_score),
                    "score_checkpoint": float(periodic_checkpoint_score),
                    "metrics": periodic_metrics,
                    "curriculum_event": curriculum_event,
                    "entropy_event": entropy_event,
                    "entropy_stage": entropy_annealer.current_stage.name,
                    "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                    "alpha": float(agent.alpha),
                    "active_distill_lambda": float(active_distill_lambda),
                }
                if stochastic_periodic_eval is not None:
                    periodic_record["stochastic_eval_root"] = str(periodic_eval_root / stoch_label)
                periodic_eval_history.append(periodic_record)
                _progress_log(
                    f"periodic_eval ep={ep + 1}/{episodes_requested} ckpt_score={periodic_checkpoint_score:+.4f} "
                    f"curr_score={periodic_score:+.4f} "
                    f"det_success_rate={float(periodic_metrics.get('det_success_rate', 0.0)):.3f} "
                    f"train_success_rate={float(periodic_metrics.get('train_success_rate', 0.0)):.3f} "
                    f"true_basin={float(periodic_metrics.get('true_basin_hit_rate', 0.0)):.3f} "
                    f"true_outer/inner/dwell="
                    f"{float(periodic_metrics.get('true_outer_hit_rate', 0.0)):.3f}/"
                    f"{float(periodic_metrics.get('true_inner_hit_rate', 0.0)):.3f}/"
                    f"{float(periodic_metrics.get('true_dwell_hit_rate', 0.0)):.3f} "
                    f"det_action_l2={float(periodic_metrics.get('det_action_l2_mean', 0.0)):.4f} "
                    f"det/stoch_action={float(periodic_metrics.get('det_action_l2_over_stoch_action_l2', 0.0)):.3f} "
                    f"entropy_stage={entropy_annealer.current_stage.name} "
                    f"target_curriculum={target_curriculum.current_stage.name}"
                )
                if periodic_score > best_eval_score + 1e-9:
                    best_eval_score = float(periodic_score)
                    best_eval_episode = int(ep)
                if periodic_checkpoint_score > best_checkpoint_score + 1e-9:
                    best_checkpoint_score = float(periodic_checkpoint_score)
                    best_checkpoint_episode = int(ep)
                    best_checkpoint_metadata = {
                        "checkpoint_kind": "best_det",
                        "episode": int(ep),
                        "periodic_eval_root": str(periodic_eval_root),
                        "action_stage_name": str(curriculum.current_stage.name),
                        "action_curriculum_max_stage": int(curriculum.max_stage_index),
                        "target_curriculum_stage_name": str(target_curriculum.current_stage.name),
                        "target_curriculum_max_stage": int(target_curriculum.max_stage_index),
                        "resolved_target_mode": str(fixed_eval_suite.get("resolved_target_mode", target_mode)),
                        "eval_suite": dict(fixed_eval_suite),
                        "exploration_std_scale": float(current_exploration_std_scale),
                        "action_scale": float(action_scale),
                        "entropy_stage": entropy_annealer.current_stage.name,
                        "target_entropy": float(entropy_annealer.current_stage.target_entropy),
                        "alpha": float(agent.alpha),
                        "entropy_annealing": entropy_annealer.to_artifact(),
                        "periodic_metrics": dict(periodic_metrics),
                        "reward_profile": str(reward_profile),
                        "reward_config": asdict(reward_composer.config),
                        "actor_mu_limit": float(actor_mu_limit),
                        "actor_bc_lambda": float(actor_bc_lambda),
                        "actor_distill_lambda": float(actor_distill_lambda),
                        "actor_distill_interval": int(actor_distill_interval),
                        "actor_distill_min_good_count": int(actor_distill_min_good_count),
                        "actor_distill_advantage_beta": float(actor_distill_advantage_beta),
                    }
                    best_checkpoint_path = _save_agent_checkpoint(
                        agent,
                        checkpoint_layout["best"],
                        run_id,
                        metadata=best_checkpoint_metadata,
                    )
                new_scale, schedule_reason = _maybe_schedule_exploration_scale(
                    bool(disable_exploration_schedule),
                    current_exploration_std_scale,
                    total_successes=successes,
                    best_min_dpos=global_best_min_dpos,
                    det_success_rate=float(periodic_metrics.get("det_success_rate", 0.0)),
                )
                if new_scale < current_exploration_std_scale - 1e-9:
                    current_exploration_std_scale = float(new_scale)
                    schedule_event = {
                        "episode": int(ep),
                        "new_exploration_std_scale": float(current_exploration_std_scale),
                        "reason": str(schedule_reason),
                    }
                    exploration_schedule_history.append(schedule_event)
                    _progress_log(
                        f"exploration_schedule ep={ep + 1}/{episodes_requested} "
                        f"new_scale={current_exploration_std_scale:.3f} reason={schedule_reason}"
                    )
                should_resume_best = (
                    best_checkpoint_path is not None
                    and int(target_curriculum.state.no_improvement_evals) >= max(1, int(resume_best_patience_evals))
                    and int(best_resume_count) < max(0, int(max_best_resume_count))
                    and int(best_checkpoint_episode) >= 0
                    and int(ep) > int(best_checkpoint_episode)
                    and int(last_best_resume_episode) != int(ep)
                )
                if should_resume_best:
                    _load_agent_checkpoint(agent, Path(best_checkpoint_path))
                    entropy_annealer.apply_to_agent(agent)
                    active_distill_lambda = (
                        float(actor_distill_lambda)
                        if (
                            not entropy_annealer.enabled
                            or int(entropy_annealer.state.stage_index) >= max(0, int(distill_start_entropy_stage_index))
                        )
                        else 0.0
                    )
                    if hasattr(agent, "set_distill_mode"):
                        agent.set_distill_mode(
                            lambda_value=active_distill_lambda,
                            stage_name=entropy_annealer.current_stage.name,
                        )
                    best_resume_count += 1
                    last_best_resume_episode = int(ep)
                    target_curriculum.state.no_improvement_evals = 0
                    if current_exploration_std_scale > 0.45:
                        current_exploration_std_scale = 0.45
                    exploration_schedule_history.append(
                        {
                            "episode": int(ep),
                            "new_exploration_std_scale": float(current_exploration_std_scale),
                            "reason": "resume_best_checkpoint",
                        }
                    )
                    _progress_log(
                        f"resume_best ep={ep + 1}/{episodes_requested} checkpoint={best_checkpoint_path} "
                        f"resume_count={best_resume_count} exploration_std_scale={current_exploration_std_scale:.3f}"
                    )
                if int(target_curriculum.state.no_improvement_evals) >= max(1, int(early_stop_patience_evals)):
                    early_stopped = True
                    early_stop_reason = (
                        f"no periodic eval improvement for {int(target_curriculum.state.no_improvement_evals)} evals"
                    )
                    _progress_log(
                        f"early_stop ep={ep + 1}/{episodes_requested} reason={early_stop_reason}"
                    )
                    break

        should_run_post_or_gap_eval = (
            runtime_mode == "gz"
            and runtime is not None
            and runtime_controlled_indices is not None
            and (int(post_train_eval_episodes) > 0 or bool(gap_eval_specs))
        )
        if should_run_post_or_gap_eval:
            post_eval_stage = curriculum.current_stage
            post_eval_fixed_suite = dict(fixed_eval_suite)
            if best_checkpoint_path:
                _load_agent_checkpoint(agent, Path(best_checkpoint_path))
                best_checkpoint_metadata = _read_agent_checkpoint_metadata(Path(best_checkpoint_path))
                if best_checkpoint_metadata.get("action_stage_name"):
                    post_eval_stage = _curriculum_stage_by_name(
                        curriculum,
                        str(best_checkpoint_metadata.get("action_stage_name")),
                    )
                if isinstance(best_checkpoint_metadata.get("eval_suite"), dict):
                    post_eval_fixed_suite = dict(best_checkpoint_metadata.get("eval_suite") or fixed_eval_suite)
                _progress_log(f"eval_resume_best checkpoint={best_checkpoint_path}")
            common_eval_kwargs = {
                "agent": agent,
                "runtime": runtime,
                "stage": post_eval_stage,
                "target_mode": str(post_eval_fixed_suite.get("resolved_target_mode", target_mode)),
                "near_home_profile": str(post_eval_fixed_suite.get("near_home_profile", near_home_profile)),
                "near_home_pos_offset_min_m": float(
                    post_eval_fixed_suite.get("near_home_pos_offset_min_m", near_home_pos_offset_min_m)
                ),
                "near_home_pos_offset_max_m": float(
                    post_eval_fixed_suite.get("near_home_pos_offset_max_m", near_home_pos_offset_max_m)
                ),
                "near_home_ori_offset_min_deg": float(
                    post_eval_fixed_suite.get("near_home_ori_offset_min_deg", near_home_ori_offset_min_deg)
                ),
                "near_home_ori_offset_max_deg": float(
                    post_eval_fixed_suite.get("near_home_ori_offset_max_deg", near_home_ori_offset_max_deg)
                ),
                "external_ee_target": external_ee_target,
                "external_ee_target_source": external_ee_target_source,
                "runtime_controlled_indices": runtime_controlled_indices,
                "policy_mode": policy_mode,
                "runtime_mode": runtime_mode,
                "ee_pos_success_threshold": ee_pos_success_threshold,
                "ee_ori_success_threshold": ee_ori_success_threshold,
                "reset_near_home_eps": reset_near_home_eps,
                "sac_seed": sac_seed,
                "gz_world_name": gz_world_name,
                "gz_target_entity_name": gz_target_entity_name,
                "gz_target_world_offset_xyz": (
                    float(gz_target_world_offset_x),
                    float(gz_target_world_offset_y),
                    float(gz_target_world_offset_z),
                ),
                "reward_config": reward_composer.config,
                "fixed_eval_suite": post_eval_fixed_suite,
            }
            if int(post_train_eval_episodes) > 0:
                eval_outputs = _run_post_training_eval_gz(
                    run_id=run_id,
                    artifact_root=artifact_root,
                    eval_root=None,
                    episodes=int(post_train_eval_episodes),
                    steps_per_episode=int(eval_step_budget),
                    **common_eval_kwargs,
                )
                post_train_eval_summary = dict(eval_outputs["summary"])
                post_train_eval_artifacts = dict(eval_outputs["artifacts"])
            if gap_eval_specs:
                gap_episodes = int(gap_eval_episodes) if int(gap_eval_episodes) > 0 else int(post_train_eval_episodes)
                gap_steps = (
                    int(gap_eval_steps_per_episode)
                    if gap_eval_steps_per_episode is not None
                    else int(eval_step_budget)
                )
                if gap_episodes > 0:
                    gap_outputs = _run_gap_diagnosis_gz(
                        run_id=run_id,
                        artifact_root=artifact_root,
                        episodes=int(gap_episodes),
                        steps_per_episode=int(gap_steps),
                        eval_specs=gap_eval_specs,
                        **common_eval_kwargs,
                    )
                    gap_eval_summary = dict(gap_outputs["summary"])
                    gap_eval_artifacts = dict(gap_outputs["artifacts"])
    finally:
        if runtime is not None:
            runtime.close()

    metrics = {
        "episodes_requested": episodes_requested,
        "episodes_completed": len(episode_outputs),
        "success_rate": _safe_rate(successes, len(episode_outputs)),
        "intervention_rate": _safe_rate(interventions, len(episode_outputs)),
        "success_rate_first": success_series[0] if success_series else 0.0,
        "success_rate_last": success_series[-1] if success_series else 0.0,
        "intervention_rate_first": intervention_series[0] if intervention_series else 0.0,
        "intervention_rate_last": intervention_series[-1] if intervention_series else 0.0,
        "reset_failures": reset_failures,
        "log_lines_expected_per_layer": expected_log_lines_per_layer,
        "reward_mean": float(np.mean(reward_totals)) if reward_totals else 0.0,
        "reward_std": float(np.std(reward_totals)) if reward_totals else 0.0,
        "reward_min": float(np.min(reward_totals)) if reward_totals else 0.0,
        "reward_max": float(np.max(reward_totals)) if reward_totals else 0.0,
    }
    if episode_outputs:
        metrics["reject_rate"] = float(
            np.mean([float(ep.get("reject_rate", 0.0)) for ep in episode_outputs])
        )
        metrics["reject_count_sum"] = float(np.sum([float(ep.get("reject_count", 0.0)) for ep in episode_outputs]))
        metrics["projection_count_sum"] = float(np.sum([float(ep.get("projection_count", 0.0)) for ep in episode_outputs]))
        metrics["near_goal_shell_count_sum"] = float(np.sum([float(ep.get("near_goal_shell_count", 0.0)) for ep in episode_outputs]))
        metrics["inner_shell_count_sum"] = float(np.sum([float(ep.get("inner_shell_count", 0.0)) for ep in episode_outputs]))
        metrics["near_goal_exit_count_sum"] = float(np.sum([float(ep.get("near_goal_exit_count", 0.0)) for ep in episode_outputs]))
        metrics["zone_exit_count_sum"] = float(np.sum([float(ep.get("zone_exit_count", 0.0)) for ep in episode_outputs]))
        metrics["shell_hit_rate"] = float(np.mean([1.0 if ep.get("shell_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["inner_shell_hit_rate"] = float(np.mean([1.0 if ep.get("inner_shell_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["dwell_hit_rate"] = float(np.mean([1.0 if ep.get("dwell_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["true_outer_hit_rate"] = float(np.mean([1.0 if ep.get("true_outer_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["true_inner_hit_rate"] = float(np.mean([1.0 if ep.get("true_inner_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["true_dwell_hit_rate"] = float(np.mean([1.0 if ep.get("true_dwell_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["true_basin_hit_rate"] = float(np.mean([1.0 if ep.get("true_basin_hit", False) else 0.0 for ep in episode_outputs]))
        metrics["true_final_outer_rate"] = float(np.mean([1.0 if ep.get("true_final_outer", False) else 0.0 for ep in episode_outputs]))
        metrics["true_final_inner_rate"] = float(np.mean([1.0 if ep.get("true_final_inner", False) else 0.0 for ep in episode_outputs]))
        metrics["true_final_dwell_rate"] = float(np.mean([1.0 if ep.get("true_final_dwell", False) else 0.0 for ep in episode_outputs]))
        metrics["true_final_basin_rate"] = float(np.mean([1.0 if ep.get("true_final_basin", False) else 0.0 for ep in episode_outputs]))
        metrics["drift_sum_total"] = float(np.sum([float(ep.get("drift_sum", 0.0)) for ep in episode_outputs]))
        metrics["sum_delta_norm"] = float(np.sum([float(ep.get("sum_delta_norm", 0.0)) for ep in episode_outputs]))
        metrics["sum_adjust_penalty"] = float(np.sum([float(ep.get("sum_adjust_penalty", 0.0)) for ep in episode_outputs]))
        metrics["sum_raw_action_penalty"] = float(np.sum([float(ep.get("sum_raw_action_penalty", 0.0)) for ep in episode_outputs]))
        metrics["sum_reject_penalty"] = float(np.sum([float(ep.get("sum_reject_penalty", 0.0)) for ep in episode_outputs]))
        metrics["final_dpos_minus_min_dpos_mean"] = float(
            np.mean([float(ep.get("final_dpos_minus_min_dpos", 0.0)) for ep in episode_outputs])
        )
    metrics["periodic_eval_interval"] = int(periodic_eval_interval)
    metrics["periodic_eval_episodes"] = int(periodic_eval_episodes)
    metrics["gap_eval_enabled"] = bool(gap_eval_specs)
    metrics["gap_eval_scales"] = [float(spec["exploration_std_scale"]) for spec in gap_eval_specs]
    metrics["gap_eval_episodes"] = int(gap_eval_episodes) if int(gap_eval_episodes) > 0 else int(post_train_eval_episodes)
    metrics["entropy_anneal_enabled"] = bool(entropy_annealer.enabled)
    metrics["entropy_anneal_mode"] = str(entropy_annealer.state.mode)
    metrics["baseline_target_entropy"] = float(entropy_annealer.state.baseline_target_entropy)
    metrics["entropy_stage"] = str(entropy_annealer.current_stage.name)
    metrics["target_entropy"] = float(entropy_annealer.current_stage.target_entropy)
    metrics["entropy_anneal_event_count"] = int(len(entropy_annealer.history))
    metrics["entropy_anneal_max_stage_index"] = int(entropy_annealer.max_stage_index)
    metrics["distill_start_entropy_stage_index"] = int(distill_start_entropy_stage_index)
    metrics["active_distill_lambda"] = float(getattr(agent, "active_distill_lambda", 0.0))
    metrics["early_stop_patience_evals"] = int(early_stop_patience_evals)
    metrics["action_curriculum_max_stage"] = int(curriculum.max_stage_index)
    metrics["target_curriculum_max_stage"] = int(target_curriculum.max_stage_index)
    metrics["resume_best_patience_evals"] = int(resume_best_patience_evals)
    metrics["max_best_resume_count"] = int(max_best_resume_count)
    metrics["best_eval_score"] = float(best_eval_score) if best_eval_score > float("-inf") else None
    metrics["best_eval_episode"] = int(best_eval_episode) if best_eval_episode >= 0 else None
    metrics["best_checkpoint_score"] = float(best_checkpoint_score) if best_checkpoint_score > float("-inf") else None
    metrics["best_checkpoint_episode"] = int(best_checkpoint_episode) if best_checkpoint_episode >= 0 else None
    metrics["best_resume_count"] = int(best_resume_count)
    metrics["final_exploration_std_scale"] = float(current_exploration_std_scale)
    metrics["early_stopped"] = bool(early_stopped)
    metrics["early_stop_reason"] = early_stop_reason
    if reset_failure_reasons:
        metrics["reset_failure_reasons"] = reset_failure_reasons[-5:]

    learning_effective = False
    ineffective_reasons: list[str] = []
    if train_metrics:
        metrics["train_actor_loss"] = float(np.mean([m["actor_loss"] for m in train_metrics]))
        metrics["train_critic_loss"] = float(np.mean([m["critic_loss"] for m in train_metrics]))
        metrics["train_alpha"] = float(train_metrics[-1]["alpha"])
        metrics["train_target_entropy"] = float(train_metrics[-1].get("target_entropy", entropy_annealer.current_stage.target_entropy))
        metrics["env_steps_collected"] = float(train_metrics[-1].get("env_steps_collected", 0.0))
        metrics["updates_applied"] = float(train_metrics[-1].get("updates_applied", 0.0))
        metrics["batch_draw_count"] = float(train_metrics[-1].get("batch_draw_count", 0.0))
        metrics["actor_update_count"] = float(train_metrics[-1].get("actor_update_count", 0.0))
        metrics["critic_update_count"] = float(train_metrics[-1].get("critic_update_count", 0.0))
        metrics["alpha_update_count"] = float(train_metrics[-1].get("alpha_update_count", 0.0))
        metrics["distill_update_count"] = float(train_metrics[-1].get("distill_update_count", 0.0))
        metrics["distill_skip_count"] = float(train_metrics[-1].get("distill_skip_count", 0.0))
        metrics["train_distill_loss"] = float(np.mean([float(m.get("distill_loss", 0.0)) for m in train_metrics]))
        metrics["train_distill_good_fraction"] = float(
            np.mean([float(m.get("distill_good_fraction", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_good_count"] = float(
            np.mean([float(m.get("distill_good_count", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_mean_action_l2"] = float(
            np.mean([float(m.get("distill_mean_action_l2", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_target_action_l2"] = float(
            np.mean([float(m.get("distill_target_action_l2", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_quality_mean"] = float(
            np.mean([float(m.get("distill_quality_mean", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_advantage_mean"] = float(
            np.mean([float(m.get("distill_advantage_mean", 0.0)) for m in train_metrics])
        )
        metrics["train_distill_active_fraction"] = float(
            np.mean([1.0 if float(m.get("distill_enabled", 0.0)) > 0.0 else 0.0 for m in train_metrics])
        )
        metrics["gradient_norm_actor"] = float(train_metrics[-1].get("gradient_norm_actor", 0.0))
        metrics["gradient_norm_critic"] = float(train_metrics[-1].get("gradient_norm_critic", 0.0))

        actor_hashes = [m.get("param_hash_actor") for m in train_metrics if m.get("param_hash_actor")]
        critic_hashes = [m.get("param_hash_critic") for m in train_metrics if m.get("param_hash_critic")]
        actor_hash_changed = len(set(actor_hashes)) > 1
        critic_hash_changed = len(set(critic_hashes)) > 1
        metrics["param_hash_actor_changed"] = actor_hash_changed
        metrics["param_hash_critic_changed"] = critic_hash_changed

        if metrics["updates_applied"] <= 0:
            ineffective_reasons.append("updates_applied<=0")
        if not actor_hash_changed:
            ineffective_reasons.append("actor_param_hash_not_changed")
        if not critic_hash_changed:
            ineffective_reasons.append("critic_param_hash_not_changed")
        learning_effective = len(ineffective_reasons) == 0
    else:
        ineffective_reasons.append("no_train_metrics")

    l1_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l1"])) for ep in episode_outputs)
    l2_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l2"])) for ep in episode_outputs)
    l3_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l3"])) for ep in episode_outputs)
    metrics.update(
        {
            "l1_log_lines": l1_lines,
            "l2_log_lines": l2_lines,
            "l3_log_lines": l3_lines,
        }
    )

    gate_result = gate_eval.evaluate(run_id=run_id, metrics=metrics)

    curriculum_path = artifact_root / "curriculum_state.json"
    gate_path = artifact_root / "gate_result.json"
    summary_path = artifact_root / "pipeline_summary.json"

    saved_checkpoint_paths: list[str] = []
    saved_checkpoint_paths.extend(entropy_checkpoint_paths)
    saved_checkpoint_paths.append(_save_agent_checkpoint(agent, checkpoint_layout["latest"], run_id))
    saved_checkpoint_paths.append(_save_agent_checkpoint(agent, checkpoint_layout["final"], run_id))
    if best_checkpoint_path is not None:
        saved_checkpoint_paths.append(best_checkpoint_path)

    curriculum_path.write_text(
        json.dumps(
            {
                "action_curriculum": curriculum.to_artifact(),
                "target_curriculum": target_curriculum.to_artifact(),
                "periodic_eval_history": periodic_eval_history,
                "early_stopped": bool(early_stopped),
                "early_stop_reason": early_stop_reason,
                "best_eval_score": (float(best_eval_score) if best_eval_score > float("-inf") else None),
                "best_eval_episode": (int(best_eval_episode) if best_eval_episode >= 0 else None),
                "best_checkpoint_score": (
                    float(best_checkpoint_score) if best_checkpoint_score > float("-inf") else None
                ),
                "best_checkpoint_episode": (
                    int(best_checkpoint_episode) if best_checkpoint_episode >= 0 else None
                ),
                "best_checkpoint_metadata": best_checkpoint_metadata,
                "best_resume_count": int(best_resume_count),
                "disable_exploration_schedule": bool(disable_exploration_schedule),
                "fixed_eval_suite": fixed_eval_suite,
                "exploration_schedule_history": exploration_schedule_history,
                "entropy_annealing": entropy_annealer.to_artifact(),
                "entropy_checkpoint_paths": list(entropy_checkpoint_paths),
                "train_metrics_path": str(train_metrics_path),
                "gap_eval_scales": [dict(spec) for spec in gap_eval_specs],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_gate_report(gate_path, gate_result)

    top_level_target_mode = episode_outputs[-1]["target_mode"] if episode_outputs else target_mode
    top_level_ee_target = episode_outputs[-1]["ee_target"] if episode_outputs else external_ee_target.tolist()
    top_level_ee_target_source = (
        episode_outputs[-1]["ee_target_source"] if episode_outputs else external_ee_target_source
    )

    summary = {
        "run_id": run_id,
        "timestamp_ns": time.time_ns(),
        "episodes": episode_outputs,
        "metrics": metrics,
        "artifacts": {
            "curriculum": str(curriculum_path),
            "gate": str(gate_path),
            "logs_root": str(logs_root),
            "reward_trace": str(reward_trace_path),
            "episode_reward_summary": str(episode_reward_summary_path),
            "runtime_trace": str(runtime_trace_path),
            "train_metrics": str(train_metrics_path),
            "checkpoint_latest": str(checkpoint_layout["latest"]),
            "checkpoint_final": str(checkpoint_layout["final"]),
        },
        "policy_mode": policy_mode,
        "runtime_mode": runtime_mode,
        "stage_profile": stage_profile,
        "target_mode": target_mode,
        "resolved_target_mode": top_level_target_mode,
        "gate_overall_decision": gate_result.overall_decision,
        "gate_passed": gate_result.overall_decision == "GO",
        "episode_joint_delta_summary": episode_joint_delta_summary,
        "ee_target": top_level_ee_target,
        "ee_target_source": top_level_ee_target_source,
        "learning_effective": learning_effective,
        "ineffective_reasons": ineffective_reasons,
        "loaded_from_checkpoint": bool(loaded_from_checkpoint),
        "loaded_checkpoint_path": loaded_checkpoint_path,
        "saved_checkpoint_paths": saved_checkpoint_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_score": (float(best_checkpoint_score) if best_checkpoint_score > float("-inf") else None),
        "best_checkpoint_episode": (int(best_checkpoint_episode) if best_checkpoint_episode >= 0 else None),
        "best_checkpoint_metadata": best_checkpoint_metadata,
        "best_resume_count": int(best_resume_count),
        "model_persistence_mode": "auto-resume-required",
        "exploration_std_scale": float(exploration_std_scale),
        "final_exploration_std_scale": float(current_exploration_std_scale),
        "disable_exploration_schedule": bool(disable_exploration_schedule),
        "action_scale": float(action_scale),
        "sac_target_entropy": float(entropy_annealer.state.baseline_target_entropy),
        "reward_profile": str(reward_profile),
        "reward_config": asdict(reward_composer.config),
        "actor_mu_limit": float(actor_mu_limit),
        "action_curriculum_max_stage": int(curriculum.max_stage_index),
        "target_curriculum_max_stage": int(target_curriculum.max_stage_index),
        "actor_update_delay": int(actor_update_delay),
        "actor_bc_lambda": float(actor_bc_lambda),
        "actor_distill_lambda": float(actor_distill_lambda),
        "actor_distill_interval": int(actor_distill_interval),
        "actor_distill_steps": int(actor_distill_steps),
        "actor_distill_batch_size": int(actor_distill_batch_size),
        "actor_distill_candidate_multiplier": int(actor_distill_candidate_multiplier),
        "actor_distill_min_good_count": int(actor_distill_min_good_count),
        "actor_distill_outer_dpos_m": float(actor_distill_outer_dpos_m),
        "actor_distill_support_dpos_m": float(actor_distill_support_dpos_m),
        "actor_distill_inner_dpos_m": float(actor_distill_inner_dpos_m),
        "actor_distill_dwell_dpos_m": float(actor_distill_dwell_dpos_m),
        "actor_distill_min_progress_m": float(actor_distill_min_progress_m),
        "actor_distill_max_delta_norm": float(actor_distill_max_delta_norm),
        "actor_distill_quality_threshold": float(actor_distill_quality_threshold),
        "actor_distill_advantage_beta": float(actor_distill_advantage_beta),
        "actor_distill_advantage_clip": float(actor_distill_advantage_clip),
        "actor_distill_grad_clip": float(actor_distill_grad_clip),
        "actor_distill_exclude_rejected": bool(actor_distill_exclude_rejected),
        "actor_distill_exclude_clamped": bool(actor_distill_exclude_clamped),
        "actor_distill_exclude_projected": bool(actor_distill_exclude_projected),
        "distill_start_entropy_stage_index": int(distill_start_entropy_stage_index),
        "target_curriculum": target_curriculum.to_artifact(),
        "fixed_eval_suite": fixed_eval_suite,
        "periodic_eval_history": periodic_eval_history,
        "exploration_schedule_history": exploration_schedule_history,
        "entropy_annealing": entropy_annealer.to_artifact(),
        "entropy_checkpoint_paths": list(entropy_checkpoint_paths),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": early_stop_reason,
        "gz_target_visualization_enabled": bool(gz_visualize_target and runtime_mode == "gz"),
        "gz_world_name": str(gz_world_name) if runtime_mode == "gz" else None,
        "gz_target_entity_name": str(gz_target_entity_name) if runtime_mode == "gz" else None,
        "gz_target_world_offset_xyz": (
            [float(gz_target_world_offset_x), float(gz_target_world_offset_y), float(gz_target_world_offset_z)]
            if runtime_mode == "gz"
            else None
        ),
        "post_train_eval": post_train_eval_summary,
        "gap_diagnosis": gap_eval_summary,
    }

    if train_metrics:
        summary["train_metrics"] = train_metrics[-20:]
    if post_train_eval_artifacts:
        summary["artifacts"].update(post_train_eval_artifacts)
    if gap_eval_artifacts:
        summary["artifacts"].update(gap_eval_artifacts)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        report_outputs = write_training_report(artifact_root)
        summary["artifacts"].update(report_outputs)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as e:
        _progress_log(f"training_report_warn run_id={run_id} error={type(e).__name__}:{e}")
        report_outputs = {}

    status = "ok"
    exit_code = 0
    if enforce_gates and gate_result.overall_decision != "GO":
        status = "gates_blocked"
        exit_code = 2

    _progress_log(
        f"run_end run_id={run_id} status={status} exit_code={exit_code} "
        f"episodes_completed={len(episode_outputs)}/{episodes_requested} "
        f"success_rate={metrics['success_rate']:.3f} intervention_rate={metrics['intervention_rate']:.3f}"
    )

    return {
        "summary": str(summary_path),
        "curriculum": str(curriculum_path),
        "gate": str(gate_path),
        "logs_root": str(logs_root),
        "reward_trace": str(reward_trace_path),
        "episode_reward_summary": str(episode_reward_summary_path),
        "runtime_trace": str(runtime_trace_path),
        **post_train_eval_artifacts,
        **gap_eval_artifacts,
        **report_outputs,
        "status": status,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5.1 minimal e2e pipeline")
    parser.add_argument("--run-id", default=f"v5_1_e2e_{int(time.time())}")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    parser.add_argument("--artifact-root", default="artifacts/v5_1/e2e")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--policy-mode", choices=["sac_torch"], default="sac_torch")
    parser.add_argument("--runtime-mode", choices=["smoke", "gz"], default="smoke")
    parser.add_argument("--stage-profile", choices=["default", "s0_b"], default="default")
    parser.add_argument("--runtime-joint-names", default="")
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--sac-seed", type=int, default=0)
    parser.add_argument("--ee-pos-success-threshold", type=float, default=0.08)
    parser.add_argument("--ee-ori-success-threshold", type=float, default=0.12)
    parser.add_argument("--external-task-prop", default="tray")
    parser.add_argument("--external-task-src-idx", type=int, default=2)
    parser.add_argument("--external-task-dst-idx", type=int, default=7)
    parser.add_argument("--external-task-waypoint-index", type=int, default=2)
    parser.add_argument("--target-mode", choices=["auto", "external", "near_home"], default="auto")
    parser.add_argument("--near-home-profile", default="s0_bootstrap")
    parser.add_argument("--near-home-pos-offset-min-m", type=float, default=0.22)
    parser.add_argument("--near-home-pos-offset-max-m", type=float, default=0.30)
    parser.add_argument("--near-home-ori-offset-min-deg", type=float, default=5.0)
    parser.add_argument("--near-home-ori-offset-max-deg", type=float, default=10.0)
    parser.add_argument("--gz-visualize-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gz-world-name", default="empty")
    parser.add_argument("--gz-target-entity-name", default="v5_1_target_marker")
    parser.add_argument("--gz-target-world-offset-x", type=float, default=0.0)
    parser.add_argument("--gz-target-world-offset-y", type=float, default=0.0)
    parser.add_argument("--gz-target-world-offset-z", type=float, default=1.04)
    parser.add_argument("--exploration-std-scale", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--sac-target-entropy", type=float, default=None)
    parser.add_argument(
        "--reward-profile",
        choices=["default", "phase_a_bootstrap", "phase_a_bootstrap_v2"],
        default="default",
    )
    parser.add_argument("--actor-mu-limit", type=float, default=1.5)
    parser.add_argument("--actor-update-delay", type=int, default=2)
    parser.add_argument("--actor-bc-lambda", type=float, default=0.05)
    parser.add_argument("--actor-bc-outer-dpos-m", type=float, default=0.08)
    parser.add_argument("--actor-bc-inner-dpos-m", type=float, default=0.04)
    parser.add_argument("--actor-bc-topk", type=int, default=3)
    parser.add_argument("--actor-distill-lambda", type=float, default=0.0)
    parser.add_argument("--actor-distill-interval", type=int, default=20)
    parser.add_argument("--actor-distill-steps", type=int, default=1)
    parser.add_argument("--actor-distill-batch-size", type=int, default=0)
    parser.add_argument("--actor-distill-candidate-multiplier", type=int, default=8)
    parser.add_argument("--actor-distill-min-good-count", type=int, default=8)
    parser.add_argument("--actor-distill-outer-dpos-m", type=float, default=0.08)
    parser.add_argument("--actor-distill-support-dpos-m", type=float, default=0.07)
    parser.add_argument("--actor-distill-inner-dpos-m", type=float, default=0.04)
    parser.add_argument("--actor-distill-dwell-dpos-m", type=float, default=0.025)
    parser.add_argument("--actor-distill-min-progress-m", type=float, default=0.003)
    parser.add_argument("--actor-distill-max-delta-norm", type=float, default=0.75)
    parser.add_argument("--actor-distill-quality-threshold", type=float, default=0.0)
    parser.add_argument("--actor-distill-advantage-beta", type=float, default=0.0)
    parser.add_argument("--actor-distill-advantage-clip", type=float, default=5.0)
    parser.add_argument("--actor-distill-grad-clip", type=float, default=1.0)
    parser.add_argument("--actor-distill-exclude-rejected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-distill-exclude-clamped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-distill-exclude-projected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-train-eval-episodes", type=int, default=5)
    parser.add_argument("--post-train-eval-steps-per-episode", type=int, default=None)
    parser.add_argument("--gap-eval-scales", default="")
    parser.add_argument("--gap-eval-episodes", type=int, default=0)
    parser.add_argument("--gap-eval-steps-per-episode", type=int, default=None)
    parser.add_argument("--entropy-anneal-mode", choices=["off", "event", "fixed"], default="off")
    parser.add_argument("--entropy-anneal-ratios", default="1.0,0.70,0.50")
    parser.add_argument("--entropy-anneal-stage-names", default="A,B,C")
    parser.add_argument("--entropy-anneal-min-episode", type=int, default=20)
    parser.add_argument("--entropy-anneal-window", type=int, default=20)
    parser.add_argument("--entropy-anneal-fixed-episodes", default="20,40")
    parser.add_argument("--entropy-anneal-max-stage-index", type=int, default=None)
    parser.add_argument("--distill-start-entropy-stage-index", type=int, default=1)
    parser.add_argument("--periodic-eval-interval", type=int, default=10)
    parser.add_argument("--periodic-eval-episodes", type=int, default=5)
    parser.add_argument("--early-stop-patience-evals", type=int, default=5)
    parser.add_argument("--action-curriculum-max-stage", type=int, default=None)
    parser.add_argument("--target-curriculum-max-stage", type=int, default=None)
    parser.add_argument("--disable-exploration-schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-best-patience-evals", type=int, default=3)
    parser.add_argument("--max-best-resume-count", type=int, default=0)
    args = parser.parse_args()

    joint_names = [x.strip() for x in args.runtime_joint_names.split(",") if x.strip()]

    outputs = run_pipeline_e2e(
        run_id=args.run_id,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        artifact_root=Path(args.artifact_root),
        enforce_gates=args.enforce_gates,
        policy_mode=args.policy_mode,
        sac_seed=args.sac_seed,
        stage_profile=args.stage_profile,
        runtime_mode=args.runtime_mode,
        runtime_joint_names=joint_names,
        trajectory_topic=args.trajectory_topic,
        joint_state_topic=args.joint_state_topic,
        ee_pos_success_threshold=args.ee_pos_success_threshold,
        ee_ori_success_threshold=args.ee_ori_success_threshold,
        external_task_prop=args.external_task_prop,
        external_task_src_idx=args.external_task_src_idx,
        external_task_dst_idx=args.external_task_dst_idx,
        external_task_waypoint_index=args.external_task_waypoint_index,
        target_mode=args.target_mode,
        near_home_profile=args.near_home_profile,
        near_home_pos_offset_min_m=args.near_home_pos_offset_min_m,
        near_home_pos_offset_max_m=args.near_home_pos_offset_max_m,
        near_home_ori_offset_min_deg=args.near_home_ori_offset_min_deg,
        near_home_ori_offset_max_deg=args.near_home_ori_offset_max_deg,
        gz_visualize_target=args.gz_visualize_target,
        gz_world_name=args.gz_world_name,
        gz_target_entity_name=args.gz_target_entity_name,
        gz_target_world_offset_x=args.gz_target_world_offset_x,
        gz_target_world_offset_y=args.gz_target_world_offset_y,
        gz_target_world_offset_z=args.gz_target_world_offset_z,
        exploration_std_scale=args.exploration_std_scale,
        action_scale=args.action_scale,
        sac_target_entropy=args.sac_target_entropy,
        reward_profile=args.reward_profile,
        actor_mu_limit=args.actor_mu_limit,
        actor_update_delay=args.actor_update_delay,
        actor_bc_lambda=args.actor_bc_lambda,
        actor_bc_outer_dpos_m=args.actor_bc_outer_dpos_m,
        actor_bc_inner_dpos_m=args.actor_bc_inner_dpos_m,
        actor_bc_topk=args.actor_bc_topk,
        actor_distill_lambda=args.actor_distill_lambda,
        actor_distill_interval=args.actor_distill_interval,
        actor_distill_steps=args.actor_distill_steps,
        actor_distill_batch_size=args.actor_distill_batch_size,
        actor_distill_candidate_multiplier=args.actor_distill_candidate_multiplier,
        actor_distill_min_good_count=args.actor_distill_min_good_count,
        actor_distill_outer_dpos_m=args.actor_distill_outer_dpos_m,
        actor_distill_support_dpos_m=args.actor_distill_support_dpos_m,
        actor_distill_inner_dpos_m=args.actor_distill_inner_dpos_m,
        actor_distill_dwell_dpos_m=args.actor_distill_dwell_dpos_m,
        actor_distill_min_progress_m=args.actor_distill_min_progress_m,
        actor_distill_max_delta_norm=args.actor_distill_max_delta_norm,
        actor_distill_quality_threshold=args.actor_distill_quality_threshold,
        actor_distill_advantage_beta=args.actor_distill_advantage_beta,
        actor_distill_advantage_clip=args.actor_distill_advantage_clip,
        actor_distill_grad_clip=args.actor_distill_grad_clip,
        actor_distill_exclude_rejected=args.actor_distill_exclude_rejected,
        actor_distill_exclude_clamped=args.actor_distill_exclude_clamped,
        actor_distill_exclude_projected=args.actor_distill_exclude_projected,
        post_train_eval_episodes=args.post_train_eval_episodes,
        post_train_eval_steps_per_episode=args.post_train_eval_steps_per_episode,
        gap_eval_scales=args.gap_eval_scales,
        gap_eval_episodes=args.gap_eval_episodes,
        gap_eval_steps_per_episode=args.gap_eval_steps_per_episode,
        entropy_anneal_mode=args.entropy_anneal_mode,
        entropy_anneal_ratios=args.entropy_anneal_ratios,
        entropy_anneal_stage_names=args.entropy_anneal_stage_names,
        entropy_anneal_min_episode=args.entropy_anneal_min_episode,
        entropy_anneal_window=args.entropy_anneal_window,
        entropy_anneal_fixed_episodes=args.entropy_anneal_fixed_episodes,
        entropy_anneal_max_stage_index=args.entropy_anneal_max_stage_index,
        distill_start_entropy_stage_index=args.distill_start_entropy_stage_index,
        periodic_eval_interval=args.periodic_eval_interval,
        periodic_eval_episodes=args.periodic_eval_episodes,
        early_stop_patience_evals=args.early_stop_patience_evals,
        action_curriculum_max_stage=args.action_curriculum_max_stage,
        target_curriculum_max_stage=args.target_curriculum_max_stage,
        disable_exploration_schedule=args.disable_exploration_schedule,
        resume_best_patience_evals=args.resume_best_patience_evals,
        max_best_resume_count=args.max_best_resume_count,
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return int(outputs.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
