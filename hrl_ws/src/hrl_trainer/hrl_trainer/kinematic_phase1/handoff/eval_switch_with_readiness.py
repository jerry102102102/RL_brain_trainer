"""Evaluate approach->dock switching driven by a dock-readiness classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..envs.switching_wrapper import SwitchingConfig
from ..eval.eval_deterministic import _load_sb3_model
from ..eval.eval_switched import _merge_switched_env_config
from ..eval.fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable
from ..training.policy_config import (
    approach_default_config_path,
    deep_merge,
    dock_default_config_path,
    load_yaml_file,
    switch_default_config_path,
    to_env_config,
    to_eval_config,
    write_json,
)
from .handoff_features import annotate_handoff_record, feature_vector
from .readiness_model import load_readiness_model, predict_readiness_score
from .train_dock_readiness_classifier import readiness_default_config_path


def _load_readiness_config(path: str | None) -> dict:
    cfg = load_yaml_file(readiness_default_config_path())
    if path:
        cfg = deep_merge(cfg, load_yaml_file(Path(path)))
    return cfg


def _copy_obs(obs: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in obs.items()}


def evaluate_switch_with_readiness(
    *,
    approach_checkpoint: Path,
    dock_checkpoint: Path,
    readiness_model_path: Path,
    approach_config_path: Path,
    dock_config_path: Path,
    switch_config_path: Path | None,
    artifact_root: Path,
    readiness_config: dict,
    approach_algorithm: str = "ppo",
    dock_algorithm: str = "td3",
) -> dict[str, object]:
    approach_cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(approach_config_path))
    dock_cfg = deep_merge(load_yaml_file(dock_default_config_path()), load_yaml_file(dock_config_path))
    switch_cfg = load_yaml_file(switch_default_config_path())
    if switch_config_path:
        switch_cfg = deep_merge(switch_cfg, load_yaml_file(switch_config_path))
    approach_env_config = to_env_config(approach_cfg)
    dock_env_config = to_env_config(dock_cfg)
    eval_cfg = to_eval_config(
        {"eval": deep_merge(deep_merge(approach_cfg.get("eval", {}), switch_cfg.get("eval", {})), readiness_config.get("eval", {}))}
    )
    switch_settings = switch_cfg.get("switch", {})
    switch_config = SwitchingConfig(**switch_settings)
    readiness_settings = readiness_config.get("readiness", {})
    threshold = float(readiness_settings.get("threshold", 0.5))
    candidate_pos_threshold = float(readiness_settings.get("candidate_pos_threshold_m", 0.03))
    confirm_steps = int(readiness_settings.get("confirm_steps", 1))

    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    dock_model = _load_sb3_model(dock_algorithm, dock_checkpoint)
    readiness_model, normalizer, saved_threshold, metadata = load_readiness_model(readiness_model_path)
    threshold = threshold if "threshold" in readiness_settings else saved_threshold
    if approach_env_config.curriculum_config.enabled and approach_env_config.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(approach_env_config, seed=eval_cfg.suite_seed, n_episodes=eval_cfg.episodes)
        scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(
            seed=eval_cfg.suite_seed,
            n_episodes=eval_cfg.episodes,
            joint_specs=approach_env_config.joint_specs,
            start_margin_fraction=approach_env_config.start_sample_margin_fraction,
            goal_margin_fraction=approach_env_config.goal_sample_margin_fraction,
        )
        scope = "fixed_random"

    merged_env_config = _merge_switched_env_config(approach_env_config, dock_env_config)
    env = ArmKinematicEnv(config=merged_env_config)
    episode_metrics: list[dict[str, object]] = []
    all_candidate_scores: list[float] = []
    all_switch_scores: list[float] = []

    for episode in suite:
        env.set_policy_mode("approach")
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        active_mode = "approach"
        terminated = truncated = False
        step = 0
        min_position_error = float(info["position_error_norm"])
        action_norms: list[float] = []
        switch_steps: list[int] = []
        switch_scores: list[float] = []
        ready_streak = 0
        switch_back_count = 0
        candidate_count = 0

        while not (terminated or truncated):
            if active_mode == "approach":
                prev_action_mag = float(np.linalg.norm(np.asarray(obs.get("prev_action", []), dtype=float)))
                score = None
                if float(info["position_error_norm"]) <= candidate_pos_threshold:
                    record = annotate_handoff_record(
                        observation=_copy_obs(obs),
                        info=info,
                        action_magnitude=prev_action_mag,
                        rollout_id=0,
                        episode_id=episode.episode_id,
                        step=step,
                        source_policy_checkpoint=str(approach_checkpoint),
                        switch_rule_config=switch_settings,
                    )
                    score = predict_readiness_score(model=readiness_model, normalizer=normalizer, features=feature_vector(record))
                    candidate_count += 1
                    all_candidate_scores.append(score)
                ready_streak = ready_streak + 1 if score is not None and score >= threshold else 0
                if ready_streak >= confirm_steps:
                    active_mode = "dock"
                    env.set_policy_mode("dock")
                    switch_steps.append(step)
                    switch_scores.append(float(score))
                    all_switch_scores.append(float(score))
                    ready_streak = 0
            else:
                if float(info["position_error_norm"]) >= switch_config.dock_exit_pos_threshold_m:
                    active_mode = "approach"
                    env.set_policy_mode("approach")
                    switch_back_count += 1

            obs = env.current_observation()
            model = dock_model if active_mode == "dock" else approach_model
            action, _ = model.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=float)
            action_norms.append(float(np.linalg.norm(action_arr)))
            obs, _, terminated, truncated, info = env.step(action_arr)
            min_position_error = min(min_position_error, float(info["position_error_norm"]))
            step += 1

        final_pos = float(info["position_error_norm"])
        episode_metrics.append(
            {
                "episode_id": episode.episode_id,
                "success": bool(info["success"]),
                "near_goal_entry": bool(info["near_goal_hit"]),
                "docking_completion": bool(info["success"]),
                "switch_count": len(switch_steps),
                "switch_steps": switch_steps,
                "first_switch_step": switch_steps[0] if switch_steps else None,
                "switch_scores": switch_scores,
                "first_switch_score": switch_scores[0] if switch_scores else None,
                "candidate_score_count": candidate_count,
                "switch_back_count": switch_back_count,
                "final_position_error": final_pos,
                "final_orientation_error": float(info["orientation_error_norm"]),
                "final_minus_min_position_error": final_pos - min_position_error,
                "action_l2_mean": float(np.mean(action_norms)) if action_norms else 0.0,
            }
        )

    switch_counts = [m["switch_count"] for m in episode_metrics]
    first_switch_steps = [m["first_switch_step"] for m in episode_metrics if m["first_switch_step"] is not None]
    success_switch_scores = [m["first_switch_score"] for m in episode_metrics if m["first_switch_score"] is not None and m["success"]]
    failed_switch_scores = [m["first_switch_score"] for m in episode_metrics if m["first_switch_score"] is not None and not m["success"]]
    summary = {
        "episode_count": len(episode_metrics),
        "overall_success_rate": float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0,
        "near_goal_entry_rate": float(np.mean([m["near_goal_entry"] for m in episode_metrics])) if episode_metrics else 0.0,
        "docking_completion_rate": float(np.mean([m["docking_completion"] for m in episode_metrics])) if episode_metrics else 0.0,
        "switch_count_mean": float(np.mean(switch_counts)) if switch_counts else 0.0,
        "confirmed_handoff_count": int(sum(switch_counts)),
        "first_switch_step_mean": float(np.mean(first_switch_steps)) if first_switch_steps else None,
        "switch_back_count": int(sum(m["switch_back_count"] for m in episode_metrics)),
        "mean_final_position_error": float(np.mean([m["final_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_orientation_error": float(np.mean([m["final_orientation_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "mean_final_minus_min_position_error": float(np.mean([m["final_minus_min_position_error"] for m in episode_metrics])) if episode_metrics else 0.0,
        "average_action_magnitude": float(np.mean([m["action_l2_mean"] for m in episode_metrics])) if episode_metrics else 0.0,
        "readiness_threshold": threshold,
        "candidate_pos_threshold_m": candidate_pos_threshold,
        "readiness_candidate_score_count": len(all_candidate_scores),
        "readiness_candidate_score_mean": float(np.mean(all_candidate_scores)) if all_candidate_scores else None,
        "readiness_switch_score_mean": float(np.mean(all_switch_scores)) if all_switch_scores else None,
        "readiness_switch_score_success_mean": float(np.mean(success_switch_scores)) if success_switch_scores else None,
        "readiness_switch_score_failure_mean": float(np.mean(failed_switch_scores)) if failed_switch_scores else None,
        "eval_scope": scope,
        "readiness_model_metadata": metadata,
        "episode_metrics": episode_metrics,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "readiness_switched_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "readiness_switched_eval_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classifier-driven Phase 1C switched handoff.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--dock-checkpoint", required=True)
    parser.add_argument("--readiness-model", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--dock-config", required=True)
    parser.add_argument("--switch-config")
    parser.add_argument("--readiness-config")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--dock-algorithm", default="td3")
    return parser


def main() -> None:  # pragma: no cover
    args = build_arg_parser().parse_args()
    summary = evaluate_switch_with_readiness(
        approach_checkpoint=Path(args.approach_checkpoint),
        dock_checkpoint=Path(args.dock_checkpoint),
        readiness_model_path=Path(args.readiness_model),
        approach_config_path=Path(args.approach_config),
        dock_config_path=Path(args.dock_config),
        switch_config_path=Path(args.switch_config) if args.switch_config else None,
        artifact_root=Path(args.artifact_root),
        readiness_config=_load_readiness_config(args.readiness_config),
        approach_algorithm=args.approach_algorithm,
        dock_algorithm=args.dock_algorithm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
