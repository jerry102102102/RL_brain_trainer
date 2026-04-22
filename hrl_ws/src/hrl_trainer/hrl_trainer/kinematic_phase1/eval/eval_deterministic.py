"""Deterministic evaluation runner for Phase 1 policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..training.policy_config import load_yaml_file, phase1_default_config_path, to_env_config
from .fixed_eval_suite import EvalEpisodeSpec, build_curriculum_local_eval_suite, default_eval_suite, suite_to_jsonable
from .metrics import EpisodeEvalMetrics, EvalConfig, episode_metrics_to_jsonable, summarize_episode_metrics


def run_deterministic_eval(
    *,
    env_factory: Callable[[], ArmKinematicEnv],
    predict_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    suite: Sequence[EvalEpisodeSpec],
    eval_config: EvalConfig | None = None,
) -> dict[str, object]:
    cfg = eval_config or EvalConfig()
    env = env_factory()
    episode_metrics: list[EpisodeEvalMetrics] = []
    for episode in suite:
        obs, info = env.reset(options=episode.reset_options())
        terminated = False
        truncated = False
        step_actions: list[float] = []
        while not (terminated or truncated):
            action = np.asarray(predict_fn(obs), dtype=float)
            step_actions.append(float(np.linalg.norm(action)))
            obs, _, terminated, truncated, info = env.step(action)
        final_err = float(info["position_error_norm"])
        min_err = float(info["min_position_error"])
        episode_metrics.append(
            EpisodeEvalMetrics(
                episode_id=episode.episode_id,
                success=bool(info["success"]),
                pre_near_goal_hit=bool(info.get("pre_near_goal_hit", False)),
                near_goal_hit=bool(info["near_goal_hit"]),
                dwell_success=int(info["dwell_count"]) >= env.config.dwell_steps_target,
                regression=final_err > min_err + cfg.regression_tolerance_m,
                final_position_error=final_err,
                final_orientation_error=float(info["orientation_error_norm"]),
                min_position_error=min_err,
                final_minus_min_position_error=final_err - min_err,
                action_l2_mean=float(np.mean(step_actions)) if step_actions else 0.0,
            )
        )
    summary = summarize_episode_metrics(episode_metrics)
    summary.update(
        {
            "suite_seed": cfg.suite_seed,
            "episodes": cfg.episodes,
            "regression_tolerance_m": cfg.regression_tolerance_m,
            "episode_metrics": episode_metrics_to_jsonable(episode_metrics),
        }
    )
    return summary


def _load_sb3_model(algorithm: str, checkpoint_path: Path):
    try:
        from stable_baselines3 import PPO, TD3
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Stable-Baselines3 is required to evaluate a saved Phase 1 policy."
        ) from exc

    algo = algorithm.lower()
    if algo == "td3":
        return TD3.load(str(checkpoint_path))
    if algo == "ppo":
        return PPO.load(str(checkpoint_path))
    raise ValueError(f"Unsupported algorithm '{algorithm}'")


def evaluate_saved_model(
    *,
    algorithm: str,
    checkpoint_path: Path,
    artifact_root: Path,
    env_config,
    eval_config: EvalConfig,
) -> dict[str, object]:
    model = _load_sb3_model(algorithm, checkpoint_path)
    suite = default_eval_suite(env_config, seed=eval_config.suite_seed, n_episodes=eval_config.episodes)

    def env_factory() -> ArmKinematicEnv:
        return ArmKinematicEnv(config=env_config)

    def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

    summary = run_deterministic_eval(env_factory=env_factory, predict_fn=predict_fn, suite=suite, eval_config=eval_config)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "deterministic_eval_suite.json").write_text(json.dumps(suite_to_jsonable(suite), indent=2))
    (artifact_root / "deterministic_eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_curriculum_stage_model(
    *,
    algorithm: str,
    checkpoint_path: Path,
    artifact_root: Path,
    env_config,
    eval_config: EvalConfig,
    stage_index: int = 0,
) -> dict[str, object]:
    model = _load_sb3_model(algorithm, checkpoint_path)
    suite = build_curriculum_local_eval_suite(
        env_config,
        seed=eval_config.suite_seed,
        stage_index=stage_index,
        n_episodes=eval_config.episodes,
    )

    def env_factory() -> ArmKinematicEnv:
        return ArmKinematicEnv(config=env_config)

    def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)

    summary = run_deterministic_eval(env_factory=env_factory, predict_fn=predict_fn, suite=suite, eval_config=eval_config)
    summary["curriculum_stage_index"] = int(stage_index)
    summary["curriculum_stage_name"] = env_config.curriculum_config.stages[int(stage_index)].name
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "curriculum_local_eval_suite.json").write_text(json.dumps(suite_to_jsonable(suite), indent=2))
    (artifact_root / "curriculum_local_eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation for a Phase 1 SB3 model.")
    parser.add_argument("--algorithm", choices=("td3", "ppo"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=700001)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    parser = build_arg_parser()
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else phase1_default_config_path()
    cfg = load_yaml_file(config_path)
    env_cfg = to_env_config(cfg)
    summary = evaluate_saved_model(
        algorithm=args.algorithm,
        checkpoint_path=Path(args.checkpoint),
        artifact_root=Path(args.artifact_root),
        env_config=env_cfg,
        eval_config=EvalConfig(suite_seed=args.eval_seed, episodes=args.eval_episodes),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI integration
    main()
