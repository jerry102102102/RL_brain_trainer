"""Train a route-curriculum RL policy on dense q-goal waypoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .eval.eval_route_curriculum import evaluate_sequential_route
from .eval.eval_route_gate import evaluate_route_gate
from .route.reward_route import RouteRewardConfig
from .route.route_curriculum import RoutePrefixCurriculumCallback, build_prefix_stages
from .route.route_dataset import load_route_dataset
from .route.route_env import RouteEnvConfig, RouteKinematicEnv
from .route.route_observation import RouteObservationConfig
from .route.route_reset_samplers import RouteResetSamplerConfig
from .route.route_sequence_env import RouteSequenceConfig, RouteSequenceKinematicEnv
from .route.teacher_anchor import RouteTeacherAnchorCallback, TeacherAnchorConfig
from .training.callbacks import PeriodicCheckpointCallback
from .training.policy_config import (
    approach_default_config_path,
    deep_merge,
    load_yaml_file,
    ppo_default_config_path,
    repo_root,
    to_algorithm_kwargs,
    to_env_config,
    write_json,
)


def _load_config(path: str | None) -> dict[str, Any]:
    cfg = deep_merge(load_yaml_file(approach_default_config_path()), load_yaml_file(ppo_default_config_path()))
    if path:
        cfg = deep_merge(cfg, load_yaml_file(Path(path)))
    return cfg


def _route_reward_config(cfg: dict[str, Any]) -> RouteRewardConfig:
    return RouteRewardConfig(**cfg.get("route", {}).get("reward", {}))


def _route_reset_config(cfg: dict[str, Any], *, max_route_index: int) -> RouteResetSamplerConfig:
    reset_cfg = dict(cfg.get("route", {}).get("reset", {}))
    reset_cfg.setdefault("max_route_index", max_route_index)
    return RouteResetSamplerConfig(**reset_cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train route curriculum policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--route-path")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--run-id", default="route_curriculum")
    parser.add_argument("--output-dir")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:  # pragma: no cover - CLI integration
    args = build_arg_parser().parse_args()
    cfg = _load_config(args.config)
    route_cfg = cfg.get("route", {})
    route_path = Path(args.route_path or route_cfg["route_path"])
    init_checkpoint = args.init_checkpoint or route_cfg.get("init_checkpoint")
    artifact_root = Path(args.output_dir) if args.output_dir else repo_root() / "artifacts" / "kinematic_phase1" / "route_curriculum" / args.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)

    route = load_route_dataset(route_path)
    prefixes = [int(x) for x in route_cfg.get("curriculum", {}).get("prefix_stages", [20, 40, 80, 120, 180, 260, 360, len(route) - 1])]
    prefixes = [min(max(1, p), len(route) - 1) for p in prefixes]
    initial_prefix = prefixes[0]
    env_cfg = to_env_config(cfg)
    route_env_cfg = RouteEnvConfig(
        base_env_config=env_cfg,
        reset_config=_route_reset_config(cfg, max_route_index=initial_prefix),
        reward_config=_route_reward_config(cfg),
        observation_config=RouteObservationConfig(**route_cfg.get("observation", {})),
    )
    sequence_cfg = RouteSequenceConfig(**route_cfg.get("sequence", {}))

    def make_env() -> RouteKinematicEnv | RouteSequenceKinematicEnv:
        if sequence_cfg.enabled:
            return RouteSequenceKinematicEnv(route=route, config=route_env_cfg, sequence_config=sequence_cfg)
        return RouteKinematicEnv(route=route, config=route_env_cfg)

    runtime_cfg = cfg.get("training", {})
    n_envs = int(runtime_cfg.get("n_envs", 1))
    device = str(runtime_cfg.get("device", "auto"))
    vec_kind = str(runtime_cfg.get("vec_env", "dummy" if n_envs == 1 else "subproc")).lower()
    vec_cls = DummyVecEnv if vec_kind == "dummy" else SubprocVecEnv
    vec_kwargs = {} if vec_cls is DummyVecEnv else {"start_method": str(runtime_cfg.get("start_method", "fork"))}
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=cfg.get("algorithms", {}).get("ppo", {}).get("seed"), vec_env_cls=vec_cls, vec_env_kwargs=vec_kwargs)

    algo_kwargs = to_algorithm_kwargs(cfg, "ppo")
    if args.total_timesteps is not None:
        algo_kwargs["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        algo_kwargs["seed"] = args.seed
    total_timesteps = int(algo_kwargs.get("total_timesteps", 100_000))
    model_kwargs = {k: v for k, v in algo_kwargs.items() if k != "total_timesteps"}

    if init_checkpoint:
        model = PPO.load(str(init_checkpoint), env=vec_env, device=device)
        model.tensorboard_log = str(artifact_root / "tb")
        if "learning_rate" in model_kwargs:
            lr = float(model_kwargs["learning_rate"])
            model.learning_rate = lr
            model.lr_schedule = FloatSchedule(lr)
            for group in model.policy.optimizer.param_groups:
                group["lr"] = lr
        print(f"Resuming route policy from {init_checkpoint}")
    else:
        model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=str(artifact_root / "tb"), device=device, **model_kwargs)

    curriculum_cfg = route_cfg.get("curriculum", {})
    curriculum_cb = RoutePrefixCurriculumCallback(
        stages=build_prefix_stages(prefixes),
        promotion_success_rate=float(curriculum_cfg.get("promotion_success_rate", 0.75)),
        promotion_route_ready_hit_rate=float(curriculum_cfg.get("promotion_route_ready_hit_rate", 0.75)),
        promotion_orientation_hit_rate=float(curriculum_cfg.get("promotion_orientation_hit_rate", 0.75)),
        promotion_max_regression_rate=float(curriculum_cfg.get("promotion_max_regression_rate", 0.25)),
        window_episodes=int(curriculum_cfg.get("promotion_window_episodes", 256)),
        min_episodes_per_stage=int(curriculum_cfg.get("min_episodes_per_stage", 256)),
    )
    callback_items = [
        PeriodicCheckpointCallback(artifact_root / "checkpoints", save_freq=int(runtime_cfg.get("checkpoint_freq", 250_000))),
        curriculum_cb,
    ]
    teacher_anchor_cfg = TeacherAnchorConfig(**route_cfg.get("teacher_anchor", {}))
    teacher_anchor_cb = None
    if teacher_anchor_cfg.enabled:
        teacher_anchor_cb = RouteTeacherAnchorCallback(teacher_anchor_cfg)
        callback_items.append(teacher_anchor_cb)
    callbacks = CallbackList(callback_items)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=not bool(init_checkpoint))
    latest_model = artifact_root / "model_latest"
    model.save(str(latest_model))
    curriculum_summary = curriculum_cb.summary()
    write_json(artifact_root / "curriculum_history.json", curriculum_summary)

    eval_end = min(int(curriculum_summary["prefix_end_index"]), len(route) - 1)
    eval_summary = evaluate_sequential_route(
        checkpoint_path=latest_model,
        config_path=Path(args.config),
        route_path=route_path,
        artifact_root=artifact_root / "route_eval_sequential",
        start_index=1,
        end_index=eval_end,
    )
    gate_summary = {"enabled": False}
    gate_cfg = route_cfg.get("sequential_gate", {})
    if bool(gate_cfg.get("enabled", False)):
        gate_summary = evaluate_route_gate(
            checkpoint_path=latest_model,
            config_path=Path(args.config),
            route_path=route_path,
            artifact_root=artifact_root / "route_gate",
            prefixes=[int(x) for x in gate_cfg.get("prefixes", [20, 40, 80, 120, 180])],
            full_end_index=gate_cfg.get("full_end_index"),
            min_prefix120_success_rate=float(gate_cfg.get("min_prefix120_success_rate", 0.98)),
            best_full_longest_prefix=int(gate_cfg.get("best_full_longest_prefix", 170)),
            full_prefix_tolerance=int(gate_cfg.get("full_prefix_tolerance", 20)),
        )
        if bool(gate_summary.get("accepted", False)):
            src = Path(str(latest_model) + ".zip")
            dst = artifact_root / "model_sequential_gate_accepted.zip"
            if src.exists():
                shutil.copy2(src, dst)
                gate_summary["accepted_model_path"] = str(dst)
    summary = {
        "schema_version": "v5.route_curriculum.training_summary.v1",
        "run_id": args.run_id,
        "route_path": str(route_path),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "model_path": str(latest_model),
        "n_envs": n_envs,
        "device": device,
        "curriculum_summary": curriculum_summary,
        "teacher_anchor_summary": teacher_anchor_cb.summary() if teacher_anchor_cb is not None else {"enabled": False},
        "route_eval_sequential_summary": eval_summary,
        "route_gate_summary": gate_summary,
        "config": cfg,
    }
    write_json(artifact_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
