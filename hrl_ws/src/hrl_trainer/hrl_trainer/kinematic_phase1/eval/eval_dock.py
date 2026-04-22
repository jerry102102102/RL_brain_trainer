"""Dock-policy deterministic evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..training.policy_config import write_json
from .eval_deterministic import _load_sb3_model
from .fixed_eval_suite import EvalEpisodeSpec, build_dock_eval_suite, suite_to_jsonable
from .metrics import EvalConfig


def run_dock_eval(
    *,
    env_factory: Callable[[], ArmKinematicEnv],
    predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    suite: Sequence[EvalEpisodeSpec],
    eval_config: EvalConfig,
) -> dict[str, object]:
    env = env_factory()
    episode_metrics: list[dict[str, object]] = []
    for episode in suite:
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "dock"})
        terminated = False
        truncated = False
        step_actions: list[float] = []
        min_position_error = float(info["position_error_norm"])
        min_orientation_error = float(info["orientation_error_norm"])
        position_only_strict_hit = min_position_error <= env.config.termination_config.success_pos_threshold_m
        orientation_only_strict_hit = min_orientation_error <= env.config.termination_config.success_ori_threshold_rad
        strict_pose_hit = position_only_strict_hit and orientation_only_strict_hit
        strict_pose_steps = int(strict_pose_hit)
        success_steps = int(bool(info["success"]))
        while not (terminated or truncated):
            action = np.asarray(predict_fn(obs), dtype=float)
            step_actions.append(float(np.linalg.norm(action)))
            obs, _, terminated, truncated, info = env.step(action)
            min_position_error = min(min_position_error, float(info["position_error_norm"]))
            min_orientation_error = min(min_orientation_error, float(info["orientation_error_norm"]))
            position_only_strict_hit = (
                position_only_strict_hit
                or float(info["position_error_norm"]) <= env.config.termination_config.success_pos_threshold_m
            )
            orientation_only_strict_hit = (
                orientation_only_strict_hit
                or float(info["orientation_error_norm"]) <= env.config.termination_config.success_ori_threshold_rad
            )
            strict_pose_hit = strict_pose_hit or (
                float(info["position_error_norm"]) <= env.config.termination_config.success_pos_threshold_m
                and float(info["orientation_error_norm"]) <= env.config.termination_config.success_ori_threshold_rad
            )
            strict_pose_steps += int(
                float(info["position_error_norm"]) <= env.config.termination_config.success_pos_threshold_m
                and float(info["orientation_error_norm"]) <= env.config.termination_config.success_ori_threshold_rad
            )
            success_steps += int(bool(info["success"]))
        final_error = float(info["position_error_norm"])
        success = bool(info["success"])
        dwell_success = int(info["dwell_count"]) >= env.config.dwell_steps_target
        step_count = max(int(info.get("step_count", len(step_actions))), len(step_actions), 1)
        strict_pose_final = (
            float(info["position_error_norm"]) <= env.config.termination_config.success_pos_threshold_m
            and float(info["orientation_error_norm"]) <= env.config.termination_config.success_ori_threshold_rad
        )
        episode_metrics.append(
            {
                "episode_id": episode.episode_id,
                "success": success,
                "dwell_success": dwell_success,
                "position_only_strict_hit": bool(position_only_strict_hit),
                "orientation_only_strict_hit": bool(orientation_only_strict_hit),
                "strict_pose_hit": bool(strict_pose_hit),
                "dwell_failure_after_strict_entry": bool(strict_pose_hit and not success),
                "strict_pose_final": bool(strict_pose_final),
                "strict_pose_step_fraction": float(strict_pose_steps / step_count),
                "success_step_fraction": float(success_steps / step_count),
                "regression": final_error > min_position_error + eval_config.regression_tolerance_m,
                "final_position_error": final_error,
                "final_orientation_error": float(info["orientation_error_norm"]),
                "min_position_error": min_position_error,
                "min_orientation_error": min_orientation_error,
                "final_minus_min_position_error": final_error - min_position_error,
                "action_l2_mean": float(np.mean(step_actions)) if step_actions else 0.0,
            }
        )

    summary = {
        "episode_count": len(episode_metrics),
        "strict_success_rate": float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "dwell_success_rate": float(np.mean([m["dwell_success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "position_only_strict_hit_rate": float(np.mean([m["position_only_strict_hit"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "orientation_only_strict_hit_rate": float(np.mean([m["orientation_only_strict_hit"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "strict_pose_hit_rate": float(np.mean([m["strict_pose_hit"] for m in episode_metrics])) if episode_metrics else 0.0,
        "strict_pose_final_rate": float(np.mean([m["strict_pose_final"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_strict_pose_step_fraction": float(np.mean([m["strict_pose_step_fraction"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "mean_success_step_fraction": float(np.mean([m["success_step_fraction"] for m in episode_metrics]))
        if episode_metrics
        else 0.0,
        "dwell_failure_after_strict_entry_rate": float(
            np.mean([m["dwell_failure_after_strict_entry"] for m in episode_metrics])
        )
        if episode_metrics
        else 0.0,
        "position_only_strict_hit_count": int(sum(1 for m in episode_metrics if m["position_only_strict_hit"])),
        "orientation_only_strict_hit_count": int(sum(1 for m in episode_metrics if m["orientation_only_strict_hit"])),
        "strict_pose_hit_count": int(sum(1 for m in episode_metrics if m["strict_pose_hit"])),
        "strict_pose_final_count": int(sum(1 for m in episode_metrics if m["strict_pose_final"])),
        "dwell_failure_after_strict_entry_count": int(
            sum(1 for m in episode_metrics if m["dwell_failure_after_strict_entry"])
        ),
        "mean_final_position_error": float(np.mean([m["final_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_orientation_error": float(np.mean([m["final_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_min_position_error": float(np.mean([m["min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_min_orientation_error": float(np.mean([m["min_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "regression_rate": float(np.mean([m["regression"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_minus_min_position_error": float(np.mean([m["final_minus_min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "average_action_magnitude": float(np.mean([m["action_l2_mean"] for m in episode_metrics])) if episode_metrics else 0.0,
        "suite_seed": eval_config.suite_seed,
        "episodes": eval_config.episodes,
        "episode_metrics": episode_metrics,
    }
    return summary


def evaluate_dock_saved_model(
    *,
    algorithm: str,
    checkpoint_path: Path,
    artifact_root: Path,
    env_config,
    eval_config: EvalConfig,
) -> dict[str, object]:
    model = _load_sb3_model(algorithm, checkpoint_path)
    suite = build_dock_eval_suite(env_config, seed=eval_config.suite_seed, n_episodes=eval_config.episodes)

    def env_factory() -> ArmKinematicEnv:
        env = ArmKinematicEnv(config=env_config)
        env.set_policy_mode("dock")
        return env

    def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

    summary = run_dock_eval(env_factory=env_factory, predict_fn=predict_fn, suite=suite, eval_config=eval_config)
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "dock_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "dock_eval_summary.json", summary)
    return summary
