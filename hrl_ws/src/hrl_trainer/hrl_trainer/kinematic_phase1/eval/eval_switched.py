"""Full switched evaluation with approach -> dock hysteresis logic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..envs.switching_wrapper import SwitchingConfig, TwoPolicySwitcher
from ..training.policy_config import write_json
from .eval_deterministic import _load_sb3_model
from .fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable
from .metrics import EvalConfig


def evaluate_switched_policies(
    *,
    approach_algorithm: str,
    approach_checkpoint_path: Path,
    dock_algorithm: str,
    dock_checkpoint_path: Path,
    artifact_root: Path,
    approach_env_config,
    dock_env_config,
    eval_config: EvalConfig,
    switching_config: SwitchingConfig,
    stage_index: int = 0,
) -> dict[str, object]:
    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint_path)
    dock_model = _load_sb3_model(dock_algorithm, dock_checkpoint_path)
    if approach_env_config.curriculum_config.enabled and approach_env_config.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(approach_env_config, seed=eval_config.suite_seed, stage_index=stage_index, n_episodes=eval_config.episodes)
        scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(seed=eval_config.suite_seed, n_episodes=eval_config.episodes, joint_specs=approach_env_config.joint_specs)
        scope = "fixed_random"

    merged_env_config = replace(
        approach_env_config,
        dock_reward_config=dock_env_config.dock_reward_config,
        dock_reset_config=dock_env_config.dock_reset_config,
        dock_residual_action_limit=dock_env_config.dock_residual_action_limit,
        dock_delta_q_change_limit_scale=dock_env_config.dock_delta_q_change_limit_scale,
        termination_config=dock_env_config.termination_config,
    )
    env = ArmKinematicEnv(config=merged_env_config)
    env.set_curriculum_stage(stage_index)
    episode_metrics: list[dict[str, object]] = []
    for episode in suite:
        switcher = TwoPolicySwitcher(config=switching_config)
        switcher.reset()
        env.set_policy_mode("approach")
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        terminated = False
        truncated = False
        step_index = 0
        min_position_error = float(info["position_error_norm"])
        action_norms: list[float] = []

        while not (terminated or truncated):
            active_mode = switcher.update(
                position_error_norm=float(info["position_error_norm"]),
                orientation_error_norm=float(info["orientation_error_norm"]),
                dwell_count=int(info["dwell_count"]),
                action_magnitude=float(info.get("action_l2", float("inf"))),
                min_position_error_so_far=float(info["min_position_error"]),
                step_index=step_index,
            )
            env.set_policy_mode(active_mode)
            obs = env.current_observation()
            model = dock_model if active_mode == "dock" else approach_model
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=float)
            action_norms.append(float(np.linalg.norm(action_arr)))
            obs, _, terminated, truncated, info = env.step(action_arr)
            min_position_error = min(min_position_error, float(info["position_error_norm"]))
            step_index += 1

        final_error = float(info["position_error_norm"])
        episode_metrics.append(
            {
                "episode_id": episode.episode_id,
                "success": bool(info["success"]),
                "near_goal_entry": bool(info["near_goal_hit"]),
                "docking_completion": bool(info["success"]),
                "switch_count": int(switcher.switch_count),
                "switch_steps": list(switcher.switch_steps),
                "first_switch_step": switcher.first_switch_step,
                "dock_timeout_count": int(switcher.dock_timeout_count),
                "switch_back_count": int(switcher.switch_back_count),
                "ready_to_dock_trigger_count": int(switcher.ready_to_dock_trigger_count),
                "ready_to_dock_confirmed_count": int(switcher.ready_to_dock_confirmed_count),
                "final_position_error": final_error,
                "final_orientation_error": float(info["orientation_error_norm"]),
                "final_minus_min_position_error": final_error - min_position_error,
                "action_l2_mean": float(np.mean(action_norms)) if action_norms else 0.0,
            }
        )

    switch_counts = [m["switch_count"] for m in episode_metrics]
    switch_steps = [step for item in episode_metrics for step in item["switch_steps"]]
    first_switch_steps = [m["first_switch_step"] for m in episode_metrics if m["first_switch_step"] is not None]
    summary = {
        "episode_count": len(episode_metrics),
        "overall_success_rate": float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "near_goal_entry_rate": float(np.mean([m["near_goal_entry"] for m in episode_metrics])) if episode_metrics else 0.0,
        "docking_completion_rate": float(np.mean([m["docking_completion"] for m in episode_metrics])) if episode_metrics else 0.0,
        "switch_count_mean": float(np.mean(switch_counts)) if switch_counts else 0.0,
        "switch_count_max": int(max(switch_counts)) if switch_counts else 0,
        "switch_step_mean": float(np.mean(switch_steps)) if switch_steps else None,
        "first_switch_step_mean": float(np.mean(first_switch_steps)) if first_switch_steps else None,
        "first_switch_step_min": int(min(first_switch_steps)) if first_switch_steps else None,
        "first_switch_step_max": int(max(first_switch_steps)) if first_switch_steps else None,
        "dock_timeout_count": int(sum(m["dock_timeout_count"] for m in episode_metrics)),
        "switch_back_count": int(sum(m["switch_back_count"] for m in episode_metrics)),
        "ready_to_dock_trigger_count": int(sum(m["ready_to_dock_trigger_count"] for m in episode_metrics)),
        "ready_to_dock_confirmed_count": int(sum(m["ready_to_dock_confirmed_count"] for m in episode_metrics)),
        "mean_final_position_error": float(np.mean([m["final_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_orientation_error": float(np.mean([m["final_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_minus_min_position_error": float(np.mean([m["final_minus_min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "average_action_magnitude": float(np.mean([m["action_l2_mean"] for m in episode_metrics])) if episode_metrics else 0.0,
        "eval_scope": scope,
        "curriculum_stage_index": int(stage_index),
        "curriculum_stage_name": approach_env_config.curriculum_config.stages[int(stage_index)].name if approach_env_config.curriculum_config.enabled and approach_env_config.curriculum_config.stages else "random_goal",
        "episode_metrics": episode_metrics,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "switched_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "switched_eval_summary.json", summary)
    return summary
