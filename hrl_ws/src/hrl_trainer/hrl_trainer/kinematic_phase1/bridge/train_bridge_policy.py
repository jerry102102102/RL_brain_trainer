"""Train a Phase 1C Bridge policy on real Approach handoff states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..training.callbacks import PeriodicCheckpointCallback
from ..training.policy_config import (
    bridge_default_config_path,
    deep_merge,
    default_artifact_root,
    load_yaml_file,
    ppo_default_config_path,
    to_algorithm_kwargs,
    to_env_config,
    write_json,
)
from .eval_bridge import evaluate_bridge_policy


def _require_training_dependencies() -> None:
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Bridge training requires gymnasium and stable-baselines3 in the active environment.") from exc


def _load_config(explicit_path: str | None) -> dict:
    cfg = deep_merge(load_yaml_file(bridge_default_config_path()), load_yaml_file(ppo_default_config_path()))
    if explicit_path:
        cfg = deep_merge(cfg, load_yaml_file(Path(explicit_path)))
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Phase 1C Bridge policy.")
    parser.add_argument("--config")
    parser.add_argument("--run-id", default="bridge_policy")
    parser.add_argument("--artifact-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-from", help="Optional PPO checkpoint/model path to continue training from.")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=920001)
    parser.add_argument("--dock-checkpoint", help="Optional best Dock policy for Bridge -> Dock validation.")
    parser.add_argument("--dock-algorithm", default="td3", choices=("ppo", "td3"))
    parser.add_argument("--dock-config", help="Optional Dock env config for Bridge -> Dock validation.")
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    _require_training_dependencies()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import FloatSchedule
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    args = build_arg_parser().parse_args()
    cfg = _load_config(args.config)
    env_cfg = to_env_config(cfg)
    algo_kwargs = to_algorithm_kwargs(cfg, "ppo")
    runtime_cfg = cfg.get("training", {})
    if args.total_timesteps is not None:
        algo_kwargs["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        algo_kwargs["seed"] = args.seed

    artifact_root = Path(args.artifact_root) if args.artifact_root else default_artifact_root("phase1c_bridge", args.run_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_root / "checkpoints"

    from ..envs.arm_kinematic_env import ArmKinematicEnv

    def _make_env() -> ArmKinematicEnv:
        env = ArmKinematicEnv(config=env_cfg)
        env.set_policy_mode("bridge")
        return env

    n_envs = int(runtime_cfg.get("n_envs", 1))
    device = str(runtime_cfg.get("device", "auto"))
    vec_env_kind = str(runtime_cfg.get("vec_env", "dummy" if n_envs == 1 else "subproc")).lower()
    vec_env_cls = DummyVecEnv if vec_env_kind == "dummy" else SubprocVecEnv
    vec_env_kwargs = {} if vec_env_cls is DummyVecEnv else {"start_method": str(runtime_cfg.get("start_method", "fork"))}
    vec_env = make_vec_env(_make_env, n_envs=n_envs, seed=algo_kwargs.get("seed"), vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)

    total_timesteps = int(algo_kwargs.get("total_timesteps", 100_000))
    model_kwargs = {k: v for k, v in algo_kwargs.items() if k != "total_timesteps"}
    if args.resume_from:
        model = PPO.load(str(args.resume_from), env=vec_env, device=device)
        model.tensorboard_log = str(artifact_root / "tb")
        if "learning_rate" in model_kwargs:
            learning_rate = float(model_kwargs["learning_rate"])
            model.learning_rate = learning_rate
            model.lr_schedule = FloatSchedule(learning_rate)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = learning_rate
        print(f"Resuming bridge policy from {args.resume_from}")
    else:
        model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(artifact_root / "tb"), device=device, **model_kwargs)

    callbacks = [PeriodicCheckpointCallback(checkpoints_dir, save_freq=int(runtime_cfg.get("checkpoint_freq", 10_000)))]
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), reset_num_timesteps=not bool(args.resume_from))

    latest_model = artifact_root / "model_latest"
    model.save(str(latest_model))
    dock_env_config = to_env_config(load_yaml_file(Path(args.dock_config))) if args.dock_config else None
    eval_summary = evaluate_bridge_policy(
        bridge_checkpoint=latest_model,
        bridge_algorithm="ppo",
        artifact_root=artifact_root / "bridge_eval",
        env_config=env_cfg,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        dock_checkpoint=Path(args.dock_checkpoint) if args.dock_checkpoint else None,
        dock_algorithm=args.dock_algorithm,
        dock_env_config=dock_env_config,
    )
    write_json(
        artifact_root / "training_summary.json",
        {
            "policy_type": "bridge",
            "algorithm": "ppo",
            "run_id": args.run_id,
            "config": cfg,
            "model_path": str(latest_model),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "n_envs": n_envs,
            "device": device,
            "bridge_eval_summary": eval_summary,
        },
    )
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
