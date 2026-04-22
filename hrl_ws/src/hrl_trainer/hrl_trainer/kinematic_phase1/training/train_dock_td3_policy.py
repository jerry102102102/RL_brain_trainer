"""Train the Phase 1B dock policy with TD3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.eval_dock import evaluate_dock_saved_model
from .callbacks import DockReverseCurriculumCallback, PeriodicCheckpointCallback
from .policy_config import (
    deep_merge,
    default_artifact_root,
    dock_default_config_path,
    load_yaml_file,
    td3_default_config_path,
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
        raise RuntimeError("Dock TD3 training requires gymnasium and stable-baselines3 in the active environment.") from exc


def _load_config(explicit_path: str | None) -> dict:
    cfg = deep_merge(load_yaml_file(dock_default_config_path()), load_yaml_file(td3_default_config_path()))
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Phase 1B dock policy with TD3.")
    parser.add_argument("--config")
    parser.add_argument("--run-id", default="dock_td3_policy")
    parser.add_argument("--artifact-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", help="Optional TD3 checkpoint/model path to continue training from.")
    return parser


def _build_td3_action_noise(algo_kwargs: dict[str, Any], action_dim: int):
    from stable_baselines3.common.noise import NormalActionNoise

    std = algo_kwargs.pop("exploration_noise_std", None)
    if std is None:
        return None
    sigma = np.full(action_dim, float(std), dtype=np.float32)
    return NormalActionNoise(mean=np.zeros(action_dim, dtype=np.float32), sigma=sigma)


def main() -> None:  # pragma: no cover - CLI integration
    _require_training_dependencies()
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = _load_config(args.config)
    env_cfg = to_env_config(cfg)
    eval_cfg = to_eval_config(cfg)
    algo_kwargs = to_algorithm_kwargs(cfg, "td3")
    runtime_cfg = cfg.get("training", {})

    if args.total_timesteps is not None:
        algo_kwargs["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        algo_kwargs["seed"] = args.seed

    artifact_root = Path(args.artifact_root) if args.artifact_root else default_artifact_root("phase1b_dock_td3", args.run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_root / "checkpoints"

    from ..envs.arm_kinematic_env import ArmKinematicEnv

    def _make_env() -> ArmKinematicEnv:
        return ArmKinematicEnv(config=env_cfg)

    n_envs = int(runtime_cfg.get("n_envs", 1))
    device = str(runtime_cfg.get("device", "auto"))
    vec_env_kind = str(runtime_cfg.get("vec_env", "dummy" if n_envs == 1 else "subproc")).lower()
    vec_env_cls = DummyVecEnv if vec_env_kind == "dummy" else SubprocVecEnv
    vec_env_kwargs = {} if vec_env_cls is DummyVecEnv else {"start_method": str(runtime_cfg.get("start_method", "fork"))}
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=algo_kwargs.get("seed"), vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)

    total_timesteps = int(algo_kwargs.get("total_timesteps", 100_000))
    model_kwargs: dict[str, Any] = {k: v for k, v in algo_kwargs.items() if k != "total_timesteps"}
    model_kwargs["action_noise"] = _build_td3_action_noise(model_kwargs, env_cfg.n_joints)

    if args.resume_from:
        model = TD3.load(str(args.resume_from), env=vec_env, device=device)
        model.tensorboard_log = str(artifact_root / "tb")
        if "learning_rate" in model_kwargs:
            learning_rate = float(model_kwargs["learning_rate"])
            model.learning_rate = learning_rate
            for param_group in model.actor.optimizer.param_groups:
                param_group["lr"] = learning_rate
            for param_group in model.critic.optimizer.param_groups:
                param_group["lr"] = learning_rate
        if model_kwargs.get("action_noise") is not None:
            model.action_noise = model_kwargs["action_noise"]
        print(f"Resuming dock TD3 policy from {args.resume_from}")
    else:
        model = TD3("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(artifact_root / "tb"), device=device, **model_kwargs)

    callbacks = [PeriodicCheckpointCallback(checkpoints_dir, save_freq=int(runtime_cfg.get("checkpoint_freq", 10_000)))]
    dock_curriculum_callback = None
    dock_curriculum_cfg = runtime_cfg.get("dock_reverse_curriculum", {})
    if bool(dock_curriculum_cfg.get("enabled", False)):
        dock_curriculum_callback = DockReverseCurriculumCallback(
            stages=list(dock_curriculum_cfg.get("stages", [])),
            window_episodes=int(dock_curriculum_cfg.get("window_episodes", 100)),
        )
        callbacks.append(dock_curriculum_callback)
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), reset_num_timesteps=not bool(args.resume_from))

    latest_model = artifact_root / "model_latest"
    model.save(str(latest_model))
    eval_summary = evaluate_dock_saved_model(
        algorithm="td3",
        checkpoint_path=latest_model,
        artifact_root=artifact_root / "dock_eval",
        env_config=env_cfg,
        eval_config=eval_cfg,
    )
    write_json(
        artifact_root / "training_summary.json",
        {
            "policy_type": "dock",
            "algorithm": "td3",
            "run_id": args.run_id,
            "config": cfg,
            "model_path": str(latest_model),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "n_envs": n_envs,
            "device": device,
            "dock_reverse_curriculum_summary": dock_curriculum_callback.summary() if dock_curriculum_callback is not None else None,
            "dock_eval_summary": eval_summary,
        },
    )
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
