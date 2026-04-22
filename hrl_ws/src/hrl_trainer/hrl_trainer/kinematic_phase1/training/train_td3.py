"""Train a Phase 1 TD3 baseline with Stable-Baselines3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..eval.eval_deterministic import evaluate_curriculum_stage_model, evaluate_saved_model
from .callbacks import PeriodicCheckpointCallback
from .policy_config import (
    default_artifact_root,
    load_training_config,
    to_algorithm_kwargs,
    to_env_config,
    to_eval_config,
    write_json,
)


def _require_training_dependencies() -> None:
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Phase 1 TD3 training requires gymnasium and stable-baselines3 in the active environment."
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Phase 1 TD3 baseline.")
    parser.add_argument("--config")
    parser.add_argument("--run-id", default="td3_baseline")
    parser.add_argument("--artifact-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    _require_training_dependencies()
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_training_config("td3", args.config)
    env_cfg = to_env_config(cfg)
    eval_cfg = to_eval_config(cfg)
    algo_kwargs = to_algorithm_kwargs(cfg, "td3")
    runtime_cfg = cfg.get("training", {})

    if args.total_timesteps is not None:
        algo_kwargs["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        algo_kwargs["seed"] = args.seed

    artifact_root = Path(args.artifact_root) if args.artifact_root else default_artifact_root("td3", args.run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_root / "checkpoints"

    def _make_env() -> ArmKinematicEnv:
        return ArmKinematicEnv(config=env_cfg)

    n_envs = int(runtime_cfg.get("n_envs", 1))
    device = str(runtime_cfg.get("device", "auto"))
    vec_env_kind = str(runtime_cfg.get("vec_env", "dummy" if n_envs == 1 else "subproc")).lower()
    vec_env_cls = DummyVecEnv if vec_env_kind == "dummy" else SubprocVecEnv
    vec_env_kwargs = {} if vec_env_cls is DummyVecEnv else {"start_method": str(runtime_cfg.get("start_method", "fork"))}
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=algo_kwargs.get("seed"), vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)
    model_kwargs: dict[str, Any] = {k: v for k, v in algo_kwargs.items() if k != "total_timesteps"}
    total_timesteps = int(algo_kwargs.get("total_timesteps", 50_000))

    model = TD3("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(artifact_root / "tb"), device=device, **model_kwargs)
    callbacks = [PeriodicCheckpointCallback(checkpoints_dir, save_freq=int(runtime_cfg.get("checkpoint_freq", 10_000)))]
    curriculum_summary: dict[str, object] | None = None
    if env_cfg.curriculum_config.enabled:
        from .callbacks import PointCurriculumCallback

        curriculum_cb = PointCurriculumCallback(
            success_rate_threshold=env_cfg.curriculum_config.success_rate_threshold,
            window_episodes=env_cfg.curriculum_config.window_episodes,
            min_episodes_per_stage=env_cfg.curriculum_config.min_episodes_per_stage,
            max_stage_index=len(env_cfg.curriculum_config.stages) - 1,
        )
        callbacks.append(curriculum_cb)
    else:
        curriculum_cb = None
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    if curriculum_cb is not None:
        curriculum_summary = curriculum_cb.summary()

    latest_model = artifact_root / "model_latest"
    model.save(str(latest_model))
    curriculum_local_eval_summary = None
    deterministic_eval_scope = "fixed_random_suite"
    if env_cfg.curriculum_config.enabled and env_cfg.curriculum_config.stages:
        stage_index = int(curriculum_summary["stage_index"]) if curriculum_summary is not None else 0
        eval_summary = evaluate_curriculum_stage_model(
            algorithm="td3",
            checkpoint_path=latest_model,
            artifact_root=artifact_root / "deterministic_eval",
            env_config=env_cfg,
            eval_config=eval_cfg,
            stage_index=stage_index,
        )
        curriculum_local_eval_summary = eval_summary
        deterministic_eval_scope = "curriculum_region"
    else:
        eval_summary = evaluate_saved_model(
            algorithm="td3",
            checkpoint_path=latest_model,
            artifact_root=artifact_root / "deterministic_eval",
            env_config=env_cfg,
            eval_config=eval_cfg,
        )
    write_json(
        artifact_root / "training_summary.json",
        {
            "algorithm": "td3",
            "run_id": args.run_id,
            "config": cfg,
            "model_path": str(latest_model),
            "n_envs": n_envs,
            "device": device,
            "deterministic_eval_scope": deterministic_eval_scope,
            "curriculum_summary": curriculum_summary,
            "deterministic_eval_summary": eval_summary,
            "curriculum_local_eval_summary": curriculum_local_eval_summary,
        },
    )
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI integration
    main()
