"""Approach-policy deterministic evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..training.policy_config import write_json
from .eval_deterministic import _load_sb3_model
from .fixed_eval_suite import EvalEpisodeSpec, build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable
from .metrics import EvalConfig


def run_approach_eval(
    *,
    env_factory: Callable[[], ArmKinematicEnv],
    predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    suite: Sequence[EvalEpisodeSpec],
    eval_config: EvalConfig,
) -> dict[str, object]:
    env = env_factory()
    episode_metrics: list[dict[str, object]] = []
    for episode in suite:
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        terminated = False
        truncated = False
        step_index = 0
        min_position_error = float(info["position_error_norm"])
        min_orientation_error = float(info["orientation_error_norm"])
        min_position_step = 0
        min_orientation_step = 0
        near_goal_step = None
        pre_near_goal_step = None
        pos_only_pre_near_goal_hit = False
        pos_only_near_goal_hit = False
        ori_gated_pre_near_goal_hit = False
        ori_gated_near_goal_hit = False
        same_step_pos_and_ori_hit = False
        same_step_dock_coarse_ready_hit = False
        dock_coarse_ready_dwell_count = 0
        max_dock_coarse_ready_dwell_count = 0
        first_dock_coarse_ready_step = None
        same_step_finisher_ready_hit = False
        finisher_ready_dwell_count = 0
        max_finisher_ready_dwell_count = 0
        first_finisher_ready_step = None
        finisher_pos_only_hit = False
        finisher_ori_only_hit = False
        failed_due_to_motion = False
        action_norms: list[float] = []
        dq_norms: list[float] = []
        reward_cfg = env.config.reward_config
        term_cfg = env.config.termination_config
        while not (terminated or truncated):
            action = np.asarray(predict_fn(obs), dtype=float)
            action_norm = float(np.linalg.norm(action))
            action_norms.append(action_norm)
            obs, _, terminated, truncated, info = env.step(action)
            dq_norm = float(info.get("executed_delta_q_l2", np.linalg.norm(info["dq"])))
            dq_norms.append(dq_norm)
            step_index += 1
            pos_err = float(info["position_error_norm"])
            ori_err = float(info["orientation_error_norm"])
            if pos_err < min_position_error:
                min_position_error = pos_err
                min_position_step = step_index
            if ori_err < min_orientation_error:
                min_orientation_error = ori_err
                min_orientation_step = step_index
            if pos_err <= reward_cfg.pre_near_goal_pos_threshold_m:
                pos_only_pre_near_goal_hit = True
            if pos_err <= reward_cfg.pre_near_goal_pos_threshold_m and ori_err <= reward_cfg.near_goal_ori_threshold_rad:
                ori_gated_pre_near_goal_hit = True
            if pos_err <= reward_cfg.near_goal_pos_threshold_m:
                pos_only_near_goal_hit = True
            if pos_err <= reward_cfg.near_goal_pos_threshold_m and ori_err <= reward_cfg.near_goal_ori_threshold_rad:
                ori_gated_near_goal_hit = True
            dc_pos_ready = (
                reward_cfg.dock_coarse_ready_pos_threshold_m > 0.0
                and pos_err <= reward_cfg.dock_coarse_ready_pos_threshold_m
            )
            dc_ori_ready = (
                reward_cfg.dock_coarse_ready_ori_threshold_rad > 0.0
                and ori_err <= reward_cfg.dock_coarse_ready_ori_threshold_rad
            )
            dc_motion_ready = (
                (reward_cfg.dock_coarse_ready_action_threshold <= 0.0 or action_norm <= reward_cfg.dock_coarse_ready_action_threshold)
                and (reward_cfg.dock_coarse_ready_dq_threshold <= 0.0 or dq_norm <= reward_cfg.dock_coarse_ready_dq_threshold)
            )
            same_step_pos_and_ori_hit = same_step_pos_and_ori_hit or bool(dc_pos_ready and dc_ori_ready)
            curr_dc_ready = bool(dc_pos_ready and dc_ori_ready and dc_motion_ready)
            same_step_dock_coarse_ready_hit = same_step_dock_coarse_ready_hit or curr_dc_ready
            if dc_pos_ready and dc_ori_ready and not dc_motion_ready:
                failed_due_to_motion = True
            if curr_dc_ready:
                if first_dock_coarse_ready_step is None:
                    first_dock_coarse_ready_step = step_index
                dock_coarse_ready_dwell_count += 1
            else:
                dock_coarse_ready_dwell_count = 0
            max_dock_coarse_ready_dwell_count = max(max_dock_coarse_ready_dwell_count, dock_coarse_ready_dwell_count)
            finisher_pos_ready = (
                reward_cfg.finisher_ready_pos_threshold_m > 0.0
                and pos_err <= reward_cfg.finisher_ready_pos_threshold_m
            )
            finisher_ori_ready = (
                reward_cfg.finisher_ready_ori_threshold_rad > 0.0
                and ori_err <= reward_cfg.finisher_ready_ori_threshold_rad
            )
            finisher_motion_ready = (
                (reward_cfg.finisher_ready_action_threshold <= 0.0 or action_norm <= reward_cfg.finisher_ready_action_threshold)
                and (reward_cfg.finisher_ready_dq_threshold <= 0.0 or dq_norm <= reward_cfg.finisher_ready_dq_threshold)
            )
            finisher_pos_only_hit = finisher_pos_only_hit or bool(finisher_pos_ready)
            finisher_ori_only_hit = finisher_ori_only_hit or bool(finisher_ori_ready)
            same_step_pos_and_ori_hit = same_step_pos_and_ori_hit or bool(finisher_pos_ready and finisher_ori_ready)
            curr_finisher_ready = bool(finisher_pos_ready and finisher_ori_ready and finisher_motion_ready)
            same_step_finisher_ready_hit = same_step_finisher_ready_hit or curr_finisher_ready
            if finisher_pos_ready and finisher_ori_ready and not finisher_motion_ready:
                failed_due_to_motion = True
            if curr_finisher_ready:
                if first_finisher_ready_step is None:
                    first_finisher_ready_step = step_index
                finisher_ready_dwell_count += 1
            else:
                finisher_ready_dwell_count = 0
            max_finisher_ready_dwell_count = max(max_finisher_ready_dwell_count, finisher_ready_dwell_count)
            if pre_near_goal_step is None and bool(info["curr_in_pre_near_goal"]):
                pre_near_goal_step = step_index
            if near_goal_step is None and bool(info["curr_in_near_goal"]):
                near_goal_step = step_index
        episode_metrics.append(
            {
                "episode_id": episode.episode_id,
                "success": bool(info["success"]),
                "pre_near_goal_hit": bool(info["pre_near_goal_hit"]),
                "near_goal_hit": bool(info["near_goal_hit"]),
                "pos_only_pre_near_goal_hit": bool(pos_only_pre_near_goal_hit),
                "pos_only_near_goal_hit": bool(pos_only_near_goal_hit),
                "ori_gated_pre_near_goal_hit": bool(ori_gated_pre_near_goal_hit),
                "ori_gated_near_goal_hit": bool(ori_gated_near_goal_hit),
                "failed_due_to_orientation": bool(pos_only_near_goal_hit and not ori_gated_near_goal_hit),
                "failed_due_to_finisher_orientation": bool(finisher_pos_only_hit and not finisher_ori_only_hit),
                "failed_due_to_motion": bool(failed_due_to_motion),
                "failed_due_to_dwell": bool(same_step_dock_coarse_ready_hit and max_dock_coarse_ready_dwell_count < 2),
                "failed_due_to_finisher_dwell": bool(same_step_finisher_ready_hit and max_finisher_ready_dwell_count < 2),
                "same_step_dock_coarse_ready_hit": bool(same_step_dock_coarse_ready_hit),
                "dock_coarse_ready_dwell": bool(max_dock_coarse_ready_dwell_count >= 2),
                "same_step_finisher_ready_hit": bool(same_step_finisher_ready_hit),
                "finisher_ready_dwell": bool(max_finisher_ready_dwell_count >= 2),
                "same_step_pos_and_ori_hit": bool(same_step_pos_and_ori_hit),
                "first_dock_coarse_ready_step": first_dock_coarse_ready_step,
                "first_finisher_ready_step": first_finisher_ready_step,
                "max_consecutive_dock_coarse_ready_steps": int(max_dock_coarse_ready_dwell_count),
                "max_consecutive_finisher_ready_steps": int(max_finisher_ready_dwell_count),
                "finisher_pos_only_hit": bool(finisher_pos_only_hit),
                "finisher_ori_only_hit": bool(finisher_ori_only_hit),
                "final_position_error": float(info["position_error_norm"]),
                "final_orientation_error": float(info["orientation_error_norm"]),
                "min_position_error": min_position_error,
                "min_orientation_error": min_orientation_error,
                "timestep_of_min_position_error": int(min_position_step),
                "timestep_of_min_orientation_error": int(min_orientation_step),
                "timestep_gap_between_min_pos_and_min_ori": int(abs(min_position_step - min_orientation_step)),
                "time_to_pre_near_goal": pre_near_goal_step,
                "time_to_near_goal": near_goal_step,
                "final_dwell_count": int(info["dwell_count"]),
                "final_action_magnitude": float(action_norms[-1]) if action_norms else 0.0,
                "final_dq_norm": float(dq_norms[-1]) if dq_norms else 0.0,
                "action_l2_mean": float(np.mean(action_norms)) if action_norms else 0.0,
            }
        )

    near_goal_steps = [m["time_to_near_goal"] for m in episode_metrics if m["time_to_near_goal"] is not None]
    first_dc_ready_steps = [
        m["first_dock_coarse_ready_step"] for m in episode_metrics if m["first_dock_coarse_ready_step"] is not None
    ]
    first_finisher_ready_steps = [
        m["first_finisher_ready_step"] for m in episode_metrics if m["first_finisher_ready_step"] is not None
    ]
    summary = {
        "episode_count": len(episode_metrics),
        "success_rate": float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pre_near_goal_hit_rate": float(np.mean([m["pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "near_goal_hit_rate": float(np.mean([m["near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pos_only_pre_near_goal_hit_rate": float(np.mean([m["pos_only_pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pos_only_near_goal_hit_rate": float(np.mean([m["pos_only_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "ori_gated_pre_near_goal_hit_rate": float(np.mean([m["ori_gated_pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "ori_gated_near_goal_hit_rate": float(np.mean([m["ori_gated_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "same_step_dock_coarse_ready_hit_rate": float(
            np.mean([m["same_step_dock_coarse_ready_hit"] for m in episode_metrics])
        )
        if episode_metrics
        else 0.0,
        "dock_coarse_ready_dwell_rate": float(np.mean([m["dock_coarse_ready_dwell"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "same_step_finisher_ready_hit_rate": float(np.mean([m["same_step_finisher_ready_hit"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "finisher_ready_hit_rate": float(np.mean([m["same_step_finisher_ready_hit"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "finisher_ready_dwell_rate": float(np.mean([m["finisher_ready_dwell"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "pos_only_hit_rate": float(np.mean([m["finisher_pos_only_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "ori_only_hit_rate": float(np.mean([m["finisher_ori_only_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "same_step_pos_and_ori_hit_rate": float(np.mean([m["same_step_pos_and_ori_hit"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "failed_due_to_orientation_count": int(sum(1 for m in episode_metrics if m["failed_due_to_orientation"])),
        "failed_due_to_finisher_orientation_count": int(
            sum(1 for m in episode_metrics if m["failed_due_to_finisher_orientation"])
        ),
        "failed_due_to_motion_count": int(sum(1 for m in episode_metrics if m["failed_due_to_motion"])),
        "failed_due_to_dwell_count": int(sum(1 for m in episode_metrics if m["failed_due_to_dwell"])),
        "failed_due_to_finisher_dwell_count": int(sum(1 for m in episode_metrics if m["failed_due_to_finisher_dwell"])),
        "mean_min_position_error": float(np.mean([m["min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_position_error": float(np.mean([m["final_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_orientation_error": float(np.mean([m["final_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_min_orientation_error": float(np.mean([m["min_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_action_magnitude": float(np.mean([m["final_action_magnitude"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_dq_norm": float(np.mean([m["final_dq_norm"] for m in episode_metrics])) if episode_metrics else 0.0,
        "timestep_of_min_position_error_mean": float(np.mean([m["timestep_of_min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "timestep_of_min_orientation_error_mean": float(np.mean([m["timestep_of_min_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "timestep_gap_between_min_pos_and_min_ori_mean": float(
            np.mean([m["timestep_gap_between_min_pos_and_min_ori"] for m in episode_metrics])
        )
        if episode_metrics
        else 0.0,
        "max_consecutive_dock_coarse_ready_steps_mean": float(
            np.mean([m["max_consecutive_dock_coarse_ready_steps"] for m in episode_metrics])
        )
        if episode_metrics
        else 0.0,
        "max_consecutive_dock_coarse_ready_steps_max": int(
            max([m["max_consecutive_dock_coarse_ready_steps"] for m in episode_metrics])
        )
        if episode_metrics
        else 0,
        "max_consecutive_finisher_ready_steps_mean": float(
            np.mean([m["max_consecutive_finisher_ready_steps"] for m in episode_metrics])
        )
        if episode_metrics
        else 0.0,
        "max_consecutive_finisher_ready_steps_max": int(
            max([m["max_consecutive_finisher_ready_steps"] for m in episode_metrics])
        )
        if episode_metrics
        else 0,
        "mean_time_to_dock_coarse_ready": float(np.mean(first_dc_ready_steps)) if first_dc_ready_steps else None,
        "mean_time_to_finisher_ready": float(np.mean(first_finisher_ready_steps)) if first_finisher_ready_steps else None,
        "mean_time_to_enter_near_goal": float(np.mean(near_goal_steps)) if near_goal_steps else None,
        "average_action_magnitude": float(np.mean([m["action_l2_mean"] for m in episode_metrics])) if episode_metrics else 0.0,
        "suite_seed": eval_config.suite_seed,
        "episodes": eval_config.episodes,
        "episode_metrics": episode_metrics,
    }
    return summary


def evaluate_approach_saved_model(
    *,
    algorithm: str,
    checkpoint_path: Path,
    artifact_root: Path,
    env_config,
    eval_config: EvalConfig,
    stage_index: int = 0,
) -> dict[str, object]:
    model = _load_sb3_model(algorithm, checkpoint_path)
    if env_config.curriculum_config.enabled and env_config.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(env_config, seed=eval_config.suite_seed, stage_index=stage_index, n_episodes=eval_config.episodes)
        suite_name = "approach_curriculum_suite.json"
    else:
        suite = build_fixed_eval_suite(seed=eval_config.suite_seed, n_episodes=eval_config.episodes, joint_specs=env_config.joint_specs)
        suite_name = "approach_fixed_suite.json"

    def env_factory() -> ArmKinematicEnv:
        env = ArmKinematicEnv(config=env_config)
        env.set_curriculum_stage(stage_index)
        env.set_policy_mode("approach")
        return env

    def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

    summary = run_approach_eval(env_factory=env_factory, predict_fn=predict_fn, suite=suite, eval_config=eval_config)
    summary["curriculum_stage_index"] = int(stage_index)
    summary["curriculum_stage_name"] = (
        env_config.curriculum_config.stages[int(stage_index)].name if env_config.curriculum_config.enabled and env_config.curriculum_config.stages else "random_goal"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / suite_name, {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "approach_eval_summary.json", summary)
    return summary
