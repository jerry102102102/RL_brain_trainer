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
        near_goal_step = None
        pre_near_goal_step = None
        pos_only_pre_near_goal_hit = False
        pos_only_near_goal_hit = False
        ori_gated_pre_near_goal_hit = False
        ori_gated_near_goal_hit = False
        success_pose_only_hit = False
        action_norms: list[float] = []
        reward_cfg = env.config.reward_config
        term_cfg = env.config.termination_config
        while not (terminated or truncated):
            action = np.asarray(predict_fn(obs), dtype=float)
            action_norms.append(float(np.linalg.norm(action)))
            obs, _, terminated, truncated, info = env.step(action)
            step_index += 1
            min_position_error = min(min_position_error, float(info["position_error_norm"]))
            min_orientation_error = min(min_orientation_error, float(info["orientation_error_norm"]))
            pos_err = float(info["position_error_norm"])
            ori_err = float(info["orientation_error_norm"])
            if pos_err <= reward_cfg.pre_near_goal_pos_threshold_m:
                pos_only_pre_near_goal_hit = True
            if pos_err <= reward_cfg.pre_near_goal_pos_threshold_m and ori_err <= reward_cfg.near_goal_ori_threshold_rad:
                ori_gated_pre_near_goal_hit = True
            if pos_err <= reward_cfg.near_goal_pos_threshold_m:
                pos_only_near_goal_hit = True
            if pos_err <= reward_cfg.near_goal_pos_threshold_m and ori_err <= reward_cfg.near_goal_ori_threshold_rad:
                ori_gated_near_goal_hit = True
            if pos_err <= term_cfg.success_pos_threshold_m and (
                (not term_cfg.require_orientation) or ori_err <= term_cfg.success_ori_threshold_rad
            ):
                success_pose_only_hit = True
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
                "failed_due_to_dwell": bool(success_pose_only_hit and not bool(info["success"])),
                "final_position_error": float(info["position_error_norm"]),
                "final_orientation_error": float(info["orientation_error_norm"]),
                "min_position_error": min_position_error,
                "min_orientation_error": min_orientation_error,
                "time_to_pre_near_goal": pre_near_goal_step,
                "time_to_near_goal": near_goal_step,
                "final_dwell_count": int(info["dwell_count"]),
                "action_l2_mean": float(np.mean(action_norms)) if action_norms else 0.0,
            }
        )

    near_goal_steps = [m["time_to_near_goal"] for m in episode_metrics if m["time_to_near_goal"] is not None]
    summary = {
        "episode_count": len(episode_metrics),
        "success_rate": float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pre_near_goal_hit_rate": float(np.mean([m["pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "near_goal_hit_rate": float(np.mean([m["near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pos_only_pre_near_goal_hit_rate": float(np.mean([m["pos_only_pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "pos_only_near_goal_hit_rate": float(np.mean([m["pos_only_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "ori_gated_pre_near_goal_hit_rate": float(np.mean([m["ori_gated_pre_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "ori_gated_near_goal_hit_rate": float(np.mean([m["ori_gated_near_goal_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "failed_due_to_orientation_count": int(sum(1 for m in episode_metrics if m["failed_due_to_orientation"])),
        "failed_due_to_dwell_count": int(sum(1 for m in episode_metrics if m["failed_due_to_dwell"])),
        "mean_min_position_error": float(np.mean([m["min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_orientation_error": float(np.mean([m["final_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_min_orientation_error": float(np.mean([m["min_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
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
