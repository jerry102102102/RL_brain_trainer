"""Build a reusable buffer of real approach -> dock handoff states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..envs.switching_wrapper import SwitchingConfig, TwoPolicySwitcher
from ..eval.eval_deterministic import _load_sb3_model
from ..eval.fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable
from .policy_config import (
    approach_default_config_path,
    deep_merge,
    load_yaml_file,
    switch_default_config_path,
    to_env_config,
    to_eval_config,
    write_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a buffer of confirmed handoff states from a trained approach policy.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--approach-config")
    parser.add_argument("--switch-config")
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage-index", type=int, default=0)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    parser = build_arg_parser()
    args = parser.parse_args()

    approach_cfg = load_yaml_file(approach_default_config_path())
    if args.approach_config:
        approach_cfg = deep_merge(approach_cfg, load_yaml_file(Path(args.approach_config)))
    switch_cfg = load_yaml_file(switch_default_config_path())
    if args.switch_config:
        switch_cfg = deep_merge(switch_cfg, load_yaml_file(Path(args.switch_config)))

    env_cfg = to_env_config(approach_cfg)
    eval_cfg = to_eval_config({"eval": deep_merge(approach_cfg.get("eval", {}), switch_cfg.get("eval", {}))})
    if args.seed is not None:
        eval_cfg = type(eval_cfg)(
            suite_seed=args.seed,
            episodes=eval_cfg.episodes,
            regression_tolerance_m=eval_cfg.regression_tolerance_m,
        )
    if args.episodes is not None:
        eval_cfg = type(eval_cfg)(
            suite_seed=eval_cfg.suite_seed,
            episodes=args.episodes,
            regression_tolerance_m=eval_cfg.regression_tolerance_m,
        )

    if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(
            env_cfg,
            seed=eval_cfg.suite_seed,
            stage_index=int(args.stage_index),
            n_episodes=eval_cfg.episodes,
        )
        eval_scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(
            seed=eval_cfg.suite_seed,
            n_episodes=eval_cfg.episodes,
            joint_specs=env_cfg.joint_specs,
            start_margin_fraction=env_cfg.start_sample_margin_fraction,
            goal_margin_fraction=env_cfg.goal_sample_margin_fraction,
        )
        eval_scope = "fixed_random"

    switcher_config = SwitchingConfig(**switch_cfg.get("switch", {}))
    model = _load_sb3_model(args.approach_algorithm, Path(args.approach_checkpoint))
    env = ArmKinematicEnv(config=env_cfg)
    states: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []

    for episode in suite:
        switcher = TwoPolicySwitcher(config=switcher_config)
        switcher.reset()
        env.set_policy_mode("approach")
        obs, info = env.reset(options={**episode.reset_options(), "policy_mode": "approach"})
        terminated = False
        truncated = False
        step_index = 0
        confirmed = False

        while not (terminated or truncated):
            previous_mode = switcher.active_mode
            active_mode = switcher.update(
                position_error_norm=float(info["position_error_norm"]),
                orientation_error_norm=float(info["orientation_error_norm"]),
                dwell_count=int(info["dwell_count"]),
                action_magnitude=float(info.get("action_l2", float("inf"))),
                min_position_error_so_far=float(info["min_position_error"]),
                step_index=step_index,
            )
            if previous_mode == "approach" and active_mode == "dock":
                dock_obs = env.current_observation()
                states.append(
                    {
                        "episode_id": int(episode.episode_id),
                        "step_index": int(step_index),
                        "initial_q": np.asarray(info["q"], dtype=float).tolist(),
                        "initial_dq": np.asarray(info["dq"], dtype=float).tolist(),
                        "initial_prev_action": np.asarray(dock_obs["prev_action"], dtype=float).tolist(),
                        "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
                        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
                        "position_error_norm": float(info["position_error_norm"]),
                        "orientation_error_norm": float(info["orientation_error_norm"]),
                        "dwell_count": int(info["dwell_count"]),
                        "action_l2": float(info.get("action_l2", 0.0)),
                    }
                )
                confirmed = True
                break

            env.set_policy_mode("approach")
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=float))
            step_index += 1

        episode_summaries.append(
            {
                "episode_id": int(episode.episode_id),
                "confirmed_handoff": bool(confirmed),
                "steps_executed": int(step_index),
                "final_position_error": float(info["position_error_norm"]),
                "final_orientation_error": float(info["orientation_error_norm"]),
            }
        )

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    buffer_path = artifact_root / "handoff_state_buffer.json"
    pos_errors = [float(item["position_error_norm"]) for item in states]
    ori_errors = [float(item["orientation_error_norm"]) for item in states]
    summary = {
        "source_approach_checkpoint": str(Path(args.approach_checkpoint)),
        "source_approach_config": str(args.approach_config) if args.approach_config else None,
        "source_switch_config": str(args.switch_config) if args.switch_config else None,
        "approach_algorithm": args.approach_algorithm,
        "eval_scope": eval_scope,
        "curriculum_stage_index": int(args.stage_index),
        "curriculum_stage_name": env_cfg.curriculum_config.stages[int(args.stage_index)].name
        if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages
        else "random_goal",
        "episode_count": len(suite),
        "confirmed_handoff_count": len(states),
        "confirmed_handoff_rate": float(len(states) / len(suite)) if suite else 0.0,
        "mean_position_error": float(np.mean(pos_errors)) if pos_errors else None,
        "mean_orientation_error": float(np.mean(ori_errors)) if ori_errors else None,
        "max_position_error": float(np.max(pos_errors)) if pos_errors else None,
        "max_orientation_error": float(np.max(ori_errors)) if ori_errors else None,
        "states": states,
        "episode_summaries": episode_summaries,
    }
    write_json(artifact_root / "handoff_state_buffer_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(buffer_path, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"states", "episode_summaries"}}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
