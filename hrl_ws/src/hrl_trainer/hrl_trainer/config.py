"""Utility dataclasses used by the HRL trainer nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from rclpy.node import Node


@dataclass
class OptionConfig:
    """Represents a macro-action option for the hierarchical controller."""

    name: str
    frac: float
    tau: float
    horizon: int


@dataclass
class TrainerHyperParams:
    gamma: float
    batch_size: int
    replay_capacity: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int
    lr: float
    tau: float
    target_update: str
    target_update_interval: int
    double_dqn: bool


@dataclass
class TrainerRuntime:
    total_training_steps: int
    warmup_steps: int
    log_interval: int
    checkpoint_interval: int
    gradient_updates_per_step: int


@dataclass
class TrainerConfig:
    """Aggregated configuration for the trainer node."""

    n_joints: int
    observation_dim: int
    control_rate: float
    command_interface: str
    goal_lower: List[float]
    goal_upper: List[float]
    success_tolerance: float
    max_episode_steps: int
    option_set: List[OptionConfig]
    hyperparams: TrainerHyperParams
    runtime: TrainerRuntime
    log_dir: Path
    checkpoint_dir: Path
    option_debug_topic: str
    use_tensorboard: bool
    summary_flush_interval: int
    summary_window: int
    training_mode: str
    seed: int
    reward_clip: float

    @classmethod
    def from_node(cls, node: Node) -> "TrainerConfig":
        params = node.get_parameters(
            [
                "n_joints",
                "observation_dim",
                "control_rate",
                "command_interface",
                "goal_lower",
                "goal_upper",
                "success_tolerance",
                "max_episode_steps",
                "gamma",
                "batch_size",
                "replay_capacity",
                "eps_start",
                "eps_end",
                "eps_decay_steps",
                "lr",
                "tau",
                "target_update",
                "target_update_interval",
                "double_dqn",
                "total_training_steps",
                "warmup_steps",
                "log_interval",
                "checkpoint_interval",
                "gradient_updates_per_step",
                "option_names",
                "option_fracs",
                "option_taus",
                "option_horizons",
                "log_dir",
                "checkpoint_dir",
                "option_debug_topic",
                "use_tensorboard",
                "summary_flush_interval",
                "summary_window",
                "training_mode",
                "seed",
                "reward_clip",
            ]
        )
        values = {p.name: p.value for p in params}
        n_joints = int(values.get("n_joints", 4))
        observation_dim = int(values.get("observation_dim", 4 * n_joints))
        goal_lower = cls._ensure_length(values.get("goal_lower", []), n_joints, "goal_lower")
        goal_upper = cls._ensure_length(values.get("goal_upper", []), n_joints, "goal_upper")
        option_names = list(values.get("option_names", []))
        option_fracs = list(values.get("option_fracs", []))
        option_taus = list(values.get("option_taus", []))
        option_horizons = list(values.get("option_horizons", []))
        if not option_names:
            option_names = ["HOLD", "SMALL_STEP", "MEDIUM_STEP", "LARGE_STEP", "SETTLE"]
        if not (len(option_names) == len(option_fracs) == len(option_taus) == len(option_horizons)):
            raise ValueError("Option arrays must have matching lengths.")
        option_set = [
            OptionConfig(
                name=str(option_names[idx]),
                frac=float(option_fracs[idx]),
                tau=float(option_taus[idx]),
                horizon=int(option_horizons[idx]),
            )
            for idx in range(len(option_names))
        ]
        hyperparams = TrainerHyperParams(
            gamma=float(values.get("gamma", 0.99)),
            batch_size=int(values.get("batch_size", 64)),
            replay_capacity=int(values.get("replay_capacity", 100000)),
            eps_start=float(values.get("eps_start", 1.0)),
            eps_end=float(values.get("eps_end", 0.05)),
            eps_decay_steps=int(values.get("eps_decay_steps", 5000)),
            lr=float(values.get("lr", 5e-4)),
            tau=float(values.get("tau", 0.01)),
            target_update=str(values.get("target_update", "soft")),
            target_update_interval=int(values.get("target_update_interval", 1000)),
            double_dqn=bool(values.get("double_dqn", True)),
        )
        runtime = TrainerRuntime(
            total_training_steps=int(values.get("total_training_steps", 20000)),
            warmup_steps=int(values.get("warmup_steps", 1000)),
            log_interval=int(values.get("log_interval", 100)),
            checkpoint_interval=int(values.get("checkpoint_interval", 1000)),
            gradient_updates_per_step=int(values.get("gradient_updates_per_step", 1)),
        )
        log_dir = Path(str(values.get("log_dir", "logs")))
        checkpoint_dir = Path(str(values.get("checkpoint_dir", "checkpoints")))
        return cls(
            n_joints=n_joints,
            observation_dim=observation_dim,
            control_rate=float(values.get("control_rate", 10.0)),
            command_interface=str(values.get("command_interface", "position")),
            goal_lower=goal_lower,
            goal_upper=goal_upper,
            success_tolerance=float(values.get("success_tolerance", 0.02)),
            max_episode_steps=int(values.get("max_episode_steps", 400)),
            option_set=option_set,
            hyperparams=hyperparams,
            runtime=runtime,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            option_debug_topic=str(values.get("option_debug_topic", "/hrl/option_debug")),
            use_tensorboard=bool(values.get("use_tensorboard", True)),
            summary_flush_interval=int(values.get("summary_flush_interval", 50)),
            summary_window=int(values.get("summary_window", 200)),
            training_mode=str(values.get("training_mode", "train")),
            seed=int(values.get("seed", 1)),
            reward_clip=float(values.get("reward_clip", 20.0)),
        )

    @staticmethod
    def _ensure_length(data: List[float], expected: int, name: str) -> List[float]:
        if not data:
            return [0.0] * expected
        if len(data) != expected:
            raise ValueError(f"Parameter '{name}' expected {expected} entries but received {len(data)}")
        return [float(x) for x in data]
