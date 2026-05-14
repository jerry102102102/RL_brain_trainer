"""Configuration helpers for the Phase 1 training entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..envs.arm_kinematic_env import Phase1EnvConfig
from ..envs.curriculum import CurriculumStageConfig, PointCurriculumConfig, default_point_curriculum_stages
from ..envs.observation_builder import ObservationBuilderConfig
from ..envs.reset_samplers import DockResetConfig, RouteResetConfig
from ..envs.reward_approach import ApproachRewardConfig
from ..envs.reward_dock import DockRewardConfig
from ..envs.termination import TerminationConfig
from ..eval.metrics import EvalConfig
from ..kinematics.joint_limits import default_joint_specs
from ..bridge.bridge_reset_samplers import BridgeResetConfig
from ..bridge.reward_bridge import BridgeRewardConfig
from ..dock_coarse.reward_dock_coarse import DockCoarseRewardConfig


def _repo_root_from_file(path: Path) -> Path:
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root from training module path")


def repo_root() -> Path:
    return _repo_root_from_file(Path(__file__).resolve())


def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def phase1_default_config_path() -> Path:
    return config_dir() / "phase1_default.yaml"


def td3_default_config_path() -> Path:
    return config_dir() / "td3_default.yaml"


def ppo_default_config_path() -> Path:
    return config_dir() / "ppo_default.yaml"


def approach_default_config_path() -> Path:
    return config_dir() / "approach_default.yaml"


def dock_default_config_path() -> Path:
    return config_dir() / "dock_default.yaml"


def switch_default_config_path() -> Path:
    return config_dir() / "switch_default.yaml"


def bridge_default_config_path() -> Path:
    return config_dir() / "bridge_default.yaml"


def dock_coarse_default_config_path() -> Path:
    return config_dir() / "dock_coarse_default.yaml"


def load_yaml_file(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(algorithm: str, explicit_path: str | None = None) -> dict[str, Any]:
    base = load_yaml_file(phase1_default_config_path())
    algo_path = td3_default_config_path() if algorithm == "td3" else ppo_default_config_path()
    algo_cfg = load_yaml_file(algo_path)
    merged = deep_merge(base, algo_cfg)
    if explicit_path:
        merged = deep_merge(merged, load_yaml_file(Path(explicit_path)))
    return merged


def to_env_config(config: dict[str, Any]) -> Phase1EnvConfig:
    env_cfg = config.get("env", {})
    reward_cfg = env_cfg.get("reward", {})
    dock_reward_cfg = env_cfg.get("dock_reward", {})
    dock_coarse_cfg = config.get("dock_coarse", {})
    dock_coarse_reward_cfg = dock_coarse_cfg.get("reward", env_cfg.get("dock_coarse_reward", {}))
    dock_reset_cfg = env_cfg.get("dock_reset", {})
    route_reset_cfg = env_cfg.get("route_reset", {})
    bridge_cfg = config.get("bridge", {})
    bridge_reward_cfg = bridge_cfg.get("reward", env_cfg.get("bridge_reward", {}))
    bridge_reset_cfg = bridge_cfg.get("reset", env_cfg.get("bridge_reset", {}))
    termination_cfg = env_cfg.get("termination", {})
    observation_cfg = env_cfg.get("observation", {})
    curriculum_cfg = env_cfg.get("curriculum", {})
    stage_dicts = curriculum_cfg.get("stages")
    if stage_dicts:
        stages = tuple(
            CurriculumStageConfig(
                name=stage["name"],
                start_q=tuple(stage["start_q"]),
                goal_q=tuple(stage["goal_q"]),
                start_noise=tuple(stage.get("start_noise", [0.0] * 7)),
                goal_noise=tuple(stage.get("goal_noise", [0.0] * 7)),
            )
            for stage in stage_dicts
        )
    else:
        stages = default_point_curriculum_stages()
    return Phase1EnvConfig(
        mode_name=str(env_cfg.get("mode", "approach")),
        n_joints=int(env_cfg.get("n_joints", 7)),
        joint_specs=default_joint_specs(),
        goal_sample_margin_fraction=float(env_cfg.get("goal_sample_margin_fraction", 0.10)),
        start_sample_margin_fraction=float(env_cfg.get("start_sample_margin_fraction", 0.20)),
        action_delta_scale=float(env_cfg.get("action_delta_scale", 1.0)),
        dynamic_action_delta_scale_enabled=bool(env_cfg.get("dynamic_action_delta_scale_enabled", False)),
        dynamic_action_delta_scale_near_pos_threshold_m=float(env_cfg.get("dynamic_action_delta_scale_near_pos_threshold_m", 0.0)),
        dynamic_action_delta_scale_far_pos_threshold_m=float(env_cfg.get("dynamic_action_delta_scale_far_pos_threshold_m", 0.0)),
        dynamic_action_delta_scale_near_multiplier=float(env_cfg.get("dynamic_action_delta_scale_near_multiplier", 1.0)),
        dynamic_action_delta_scale_far_multiplier=float(env_cfg.get("dynamic_action_delta_scale_far_multiplier", 1.0)),
        dock_action_delta_scale=float(env_cfg.get("dock_action_delta_scale", 0.0)),
        dock_residual_action_limit=float(env_cfg.get("dock_residual_action_limit", 1.0)),
        dock_delta_q_change_limit_scale=float(env_cfg.get("dock_delta_q_change_limit_scale", 0.0)),
        dock_dynamic_action_limit_near_pos_threshold_m=float(env_cfg.get("dock_dynamic_action_limit_near_pos_threshold_m", 0.0)),
        dock_dynamic_action_limit_far_pos_threshold_m=float(env_cfg.get("dock_dynamic_action_limit_far_pos_threshold_m", 0.0)),
        dock_dynamic_residual_action_limit_near=float(env_cfg.get("dock_dynamic_residual_action_limit_near", env_cfg.get("dock_residual_action_limit", 1.0))),
        dock_dynamic_residual_action_limit_far=float(env_cfg.get("dock_dynamic_residual_action_limit_far", env_cfg.get("dock_residual_action_limit", 1.0))),
        dock_dynamic_delta_q_change_limit_scale_near=float(env_cfg.get("dock_dynamic_delta_q_change_limit_scale_near", env_cfg.get("dock_delta_q_change_limit_scale", 0.0))),
        dock_dynamic_delta_q_change_limit_scale_far=float(env_cfg.get("dock_dynamic_delta_q_change_limit_scale_far", env_cfg.get("dock_delta_q_change_limit_scale", 0.0))),
        episode_length=int(env_cfg.get("episode_length", 75)),
        dwell_steps_target=int(termination_cfg.get("success_dwell_steps", 3)),
        curriculum_config=PointCurriculumConfig(
            enabled=bool(curriculum_cfg.get("enabled", True)),
            success_rate_threshold=float(curriculum_cfg.get("success_rate_threshold", 0.80)),
            window_episodes=int(curriculum_cfg.get("window_episodes", 20)),
            min_episodes_per_stage=int(curriculum_cfg.get("min_episodes_per_stage", 30)),
            stages=stages,
        ),
        workspace_stage_sampling=dict(env_cfg.get("workspace_stage_sampling", {})),
        reward_config=ApproachRewardConfig(**reward_cfg),
        dock_reward_config=DockRewardConfig(**dock_reward_cfg),
        dock_coarse_reward_config=DockCoarseRewardConfig(**dock_coarse_reward_cfg),
        dock_reset_config=DockResetConfig(**dock_reset_cfg) if dock_reset_cfg else DockResetConfig(),
        route_reset_config=RouteResetConfig(**route_reset_cfg) if route_reset_cfg else RouteResetConfig(),
        bridge_reward_config=BridgeRewardConfig(**bridge_reward_cfg),
        bridge_reset_config=BridgeResetConfig(**bridge_reset_cfg) if bridge_reset_cfg else BridgeResetConfig(),
        termination_config=TerminationConfig(**termination_cfg),
        observation_config=ObservationBuilderConfig(**observation_cfg),
    )


def to_eval_config(config: dict[str, Any]) -> EvalConfig:
    eval_cfg = config.get("eval", {})
    return EvalConfig(
        suite_seed=int(eval_cfg.get("suite_seed", 700001)),
        episodes=int(eval_cfg.get("episodes", 10)),
        regression_tolerance_m=float(eval_cfg.get("regression_tolerance_m", 0.01)),
    )


def to_algorithm_kwargs(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    return dict(config.get("algorithms", {}).get(algorithm, {}))


def default_artifact_root(algorithm: str, run_id: str) -> Path:
    return repo_root() / "artifacts" / "kinematic_phase1" / algorithm / run_id


def training_runtime_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("training", {}))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
