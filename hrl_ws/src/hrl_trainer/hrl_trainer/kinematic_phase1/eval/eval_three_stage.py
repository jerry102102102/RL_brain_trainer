"""Offline Approach -> Dock-Coarse -> Dock-Finisher evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..training.policy_config import (
    approach_default_config_path,
    deep_merge,
    dock_coarse_default_config_path,
    load_yaml_file,
    to_env_config,
    write_json,
)
from .eval_deterministic import _load_sb3_model
from .fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite, suite_to_jsonable


def _predict(model: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    action, _ = model.predict(obs, deterministic=True)
    return np.asarray(action, dtype=float)


def _state_reset_options(result: dict[str, Any], *, policy_mode: str) -> dict[str, Any]:
    return {
        "initial_q": result["final_q"],
        "initial_dq": result["final_dq"],
        "initial_prev_action": result["final_prev_action"],
        "goal_q": result["goal_q"],
        "goal_pose6": result["goal_pose6"],
        "policy_mode": policy_mode,
    }


def _dock_coarse_ready(
    *,
    pos: float,
    ori: float,
    action_norm: float,
    dq_norm: float,
    cfg,
) -> bool:
    return bool(
        cfg.dock_coarse_ready_pos_threshold_m > 0.0
        and cfg.dock_coarse_ready_ori_threshold_rad > 0.0
        and pos <= cfg.dock_coarse_ready_pos_threshold_m
        and ori <= cfg.dock_coarse_ready_ori_threshold_rad
        and (cfg.dock_coarse_ready_action_threshold <= 0.0 or action_norm <= cfg.dock_coarse_ready_action_threshold)
        and (cfg.dock_coarse_ready_dq_threshold <= 0.0 or dq_norm <= cfg.dock_coarse_ready_dq_threshold)
    )


def _run_policy(
    *,
    env: ArmKinematicEnv,
    model: Any,
    reset_options: dict[str, Any],
    ready_cfg=None,
) -> dict[str, Any]:
    obs, info = env.reset(options=reset_options)
    terminated = False
    truncated = False
    steps = 0
    min_pos = float(info["position_error_norm"])
    min_ori = float(info["orientation_error_norm"])
    action_norms: list[float] = []
    dq_norms: list[float] = []
    ready_hit = False
    ready_streak = 0
    max_ready_streak = 0
    first_ready_step: int | None = None
    while not (terminated or truncated):
        action = _predict(model, obs)
        action_norm = float(np.linalg.norm(action))
        action_norms.append(action_norm)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        dq_norm = float(info.get("executed_delta_q_l2", np.linalg.norm(info["dq"])))
        dq_norms.append(dq_norm)
        pos = float(info["position_error_norm"])
        ori = float(info["orientation_error_norm"])
        min_pos = min(min_pos, pos)
        min_ori = min(min_ori, ori)
        if ready_cfg is not None and _dock_coarse_ready(
            pos=pos,
            ori=ori,
            action_norm=action_norm,
            dq_norm=dq_norm,
            cfg=ready_cfg,
        ):
            ready_hit = True
            if first_ready_step is None:
                first_ready_step = steps
            ready_streak += 1
        else:
            ready_streak = 0
        max_ready_streak = max(max_ready_streak, ready_streak)

    return {
        "success": bool(info["success"]),
        "final_position_error": float(info["position_error_norm"]),
        "final_orientation_error": float(info["orientation_error_norm"]),
        "min_position_error": min_pos,
        "min_orientation_error": min_ori,
        "final_action_magnitude": float(action_norms[-1]) if action_norms else 0.0,
        "final_dq_norm": float(dq_norms[-1]) if dq_norms else 0.0,
        "mean_action_magnitude": float(np.mean(action_norms)) if action_norms else 0.0,
        "mean_dq_norm": float(np.mean(dq_norms)) if dq_norms else 0.0,
        "dock_coarse_ready_hit": bool(ready_hit),
        "dock_coarse_ready_dwell": bool(max_ready_streak >= 2),
        "max_dock_coarse_ready_streak": int(max_ready_streak),
        "first_dock_coarse_ready_step": first_ready_step,
        "step_count": int(steps),
        "final_q": np.asarray(info["q"], dtype=float).tolist(),
        "final_dq": np.asarray(info["dq"], dtype=float).tolist(),
        "final_prev_action": np.asarray(env._prev_action, dtype=float).tolist(),  # noqa: SLF001 - state handoff eval
        "goal_q": np.asarray(info["goal_q"], dtype=float).tolist(),
        "goal_pose6": np.asarray(info["goal_pose6"], dtype=float).tolist(),
    }


def _summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"episode_count": 0}
    first_ready_steps = [
        m["approach_first_dock_coarse_ready_step"]
        for m in metrics
        if m["approach_first_dock_coarse_ready_step"] is not None
    ]
    return {
        "episode_count": len(metrics),
        "approach_dock_coarse_ready_hit_rate": float(
            np.mean([m["approach_dock_coarse_ready_hit"] for m in metrics])
        ),
        "approach_dock_coarse_ready_dwell_rate": float(
            np.mean([m["approach_dock_coarse_ready_dwell"] for m in metrics])
        ),
        "approach_mean_max_dock_coarse_ready_streak": float(
            np.mean([m["approach_max_dock_coarse_ready_streak"] for m in metrics])
        ),
        "approach_mean_first_dock_coarse_ready_step": float(np.mean(first_ready_steps)) if first_ready_steps else None,
        "approach_mean_final_position_error": float(np.mean([m["approach_final_position_error"] for m in metrics])),
        "approach_mean_final_orientation_error": float(np.mean([m["approach_final_orientation_error"] for m in metrics])),
        "approach_mean_final_action_magnitude": float(np.mean([m["approach_final_action_magnitude"] for m in metrics])),
        "approach_mean_final_dq_norm": float(np.mean([m["approach_final_dq_norm"] for m in metrics])),
        "dock_coarse_success_rate": float(np.mean([m["dock_coarse_success"] for m in metrics])),
        "dock_coarse_mean_final_position_error": float(np.mean([m["dock_coarse_final_position_error"] for m in metrics])),
        "dock_coarse_mean_final_orientation_error": float(np.mean([m["dock_coarse_final_orientation_error"] for m in metrics])),
        "finisher_success_rate": float(np.mean([m["finisher_success"] for m in metrics])),
        "finisher_mean_final_position_error": float(np.mean([m["finisher_final_position_error"] for m in metrics])),
        "finisher_mean_final_orientation_error": float(np.mean([m["finisher_final_orientation_error"] for m in metrics])),
    }


def evaluate_three_stage_pipeline(
    *,
    approach_checkpoint: Path,
    approach_algorithm: str,
    dock_coarse_checkpoint: Path,
    dock_coarse_algorithm: str,
    finisher_checkpoint: Path,
    finisher_algorithm: str,
    artifact_root: Path,
    approach_env_config: Phase1EnvConfig,
    dock_coarse_env_config: Phase1EnvConfig,
    finisher_env_config: Phase1EnvConfig,
    episodes: int,
    seed: int,
    stage_index: int = 0,
) -> dict[str, Any]:
    approach_model = _load_sb3_model(approach_algorithm, approach_checkpoint)
    dock_coarse_model = _load_sb3_model(dock_coarse_algorithm, dock_coarse_checkpoint)
    finisher_model = _load_sb3_model(finisher_algorithm, finisher_checkpoint)
    if approach_env_config.curriculum_config.enabled and approach_env_config.curriculum_config.stages:
        suite = build_curriculum_local_eval_suite(
            approach_env_config,
            seed=seed,
            stage_index=stage_index,
            n_episodes=episodes,
        )
        scope = "curriculum_region"
    else:
        suite = build_fixed_eval_suite(seed=seed, n_episodes=episodes, joint_specs=approach_env_config.joint_specs)
        scope = "fixed_random"

    metrics: list[dict[str, Any]] = []
    for episode in suite:
        approach_env = ArmKinematicEnv(config=approach_env_config)
        approach_env.set_curriculum_stage(stage_index)
        approach_env.set_policy_mode("approach")
        approach_result = _run_policy(
            env=approach_env,
            model=approach_model,
            reset_options={**episode.reset_options(), "policy_mode": "approach"},
            ready_cfg=approach_env_config.reward_config,
        )

        coarse_env = ArmKinematicEnv(config=dock_coarse_env_config)
        coarse_env.set_policy_mode("dock_coarse")
        coarse_result = _run_policy(
            env=coarse_env,
            model=dock_coarse_model,
            reset_options=_state_reset_options(approach_result, policy_mode="dock_coarse"),
        )

        finisher_env = ArmKinematicEnv(config=finisher_env_config)
        finisher_env.set_policy_mode("dock")
        finisher_result = _run_policy(
            env=finisher_env,
            model=finisher_model,
            reset_options=_state_reset_options(coarse_result, policy_mode="dock"),
        )

        metrics.append(
            {
                "episode_id": episode.episode_id,
                **{f"approach_{key}": value for key, value in approach_result.items() if not key.startswith("final_") or key in {
                    "final_position_error",
                    "final_orientation_error",
                    "final_action_magnitude",
                    "final_dq_norm",
                }},
                "approach_final_position_error": approach_result["final_position_error"],
                "approach_final_orientation_error": approach_result["final_orientation_error"],
                "approach_final_action_magnitude": approach_result["final_action_magnitude"],
                "approach_final_dq_norm": approach_result["final_dq_norm"],
                "approach_dock_coarse_ready_hit": approach_result["dock_coarse_ready_hit"],
                "approach_dock_coarse_ready_dwell": approach_result["dock_coarse_ready_dwell"],
                "approach_max_dock_coarse_ready_streak": approach_result["max_dock_coarse_ready_streak"],
                "approach_first_dock_coarse_ready_step": approach_result["first_dock_coarse_ready_step"],
                "dock_coarse_success": coarse_result["success"],
                "dock_coarse_final_position_error": coarse_result["final_position_error"],
                "dock_coarse_final_orientation_error": coarse_result["final_orientation_error"],
                "dock_coarse_final_action_magnitude": coarse_result["final_action_magnitude"],
                "dock_coarse_final_dq_norm": coarse_result["final_dq_norm"],
                "finisher_success": finisher_result["success"],
                "finisher_final_position_error": finisher_result["final_position_error"],
                "finisher_final_orientation_error": finisher_result["final_orientation_error"],
                "finisher_final_action_magnitude": finisher_result["final_action_magnitude"],
                "finisher_final_dq_norm": finisher_result["final_dq_norm"],
            }
        )

    summary = {
        "approach_checkpoint": str(approach_checkpoint),
        "dock_coarse_checkpoint": str(dock_coarse_checkpoint),
        "finisher_checkpoint": str(finisher_checkpoint),
        "episodes": int(episodes),
        "seed": int(seed),
        "eval_scope": scope,
        "curriculum_stage_index": int(stage_index),
        **_summary(metrics),
        "episode_metrics": metrics,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "three_stage_eval_suite.json", {"suite": suite_to_jsonable(suite)})
    write_json(artifact_root / "three_stage_eval_summary.json", summary)
    return summary


def _load_approach_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(approach_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def _load_dock_coarse_config(config_path: str | None) -> dict[str, Any]:
    cfg = load_yaml_file(dock_coarse_default_config_path())
    if config_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(config_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Approach -> Dock-Coarse -> Dock-Finisher.")
    parser.add_argument("--approach-checkpoint", required=True)
    parser.add_argument("--dock-coarse-checkpoint", required=True)
    parser.add_argument("--finisher-checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--approach-config")
    parser.add_argument("--dock-coarse-config")
    parser.add_argument("--finisher-config", required=True)
    parser.add_argument("--approach-algorithm", default="ppo")
    parser.add_argument("--dock-coarse-algorithm", default="ppo")
    parser.add_argument("--finisher-algorithm", default="td3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=700001)
    parser.add_argument("--stage-index", type=int, default=0)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    summary = evaluate_three_stage_pipeline(
        approach_checkpoint=Path(args.approach_checkpoint),
        approach_algorithm=args.approach_algorithm,
        dock_coarse_checkpoint=Path(args.dock_coarse_checkpoint),
        dock_coarse_algorithm=args.dock_coarse_algorithm,
        finisher_checkpoint=Path(args.finisher_checkpoint),
        finisher_algorithm=args.finisher_algorithm,
        artifact_root=Path(args.artifact_root),
        approach_env_config=to_env_config(_load_approach_config(args.approach_config)),
        dock_coarse_env_config=to_env_config(_load_dock_coarse_config(args.dock_coarse_config)),
        finisher_env_config=to_env_config(load_yaml_file(Path(args.finisher_config))),
        episodes=args.episodes,
        seed=args.seed,
        stage_index=args.stage_index,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
