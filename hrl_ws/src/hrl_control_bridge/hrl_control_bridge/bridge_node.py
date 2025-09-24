"""Configuration helpers for the HRL control bridge node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from rclpy.node import Node


@dataclass
class BridgeConfig:
    """Dataclass capturing the ROS 2 parameters for the control bridge."""

    control_frequency: float
    joint_names: List[str]
    command_topic: str
    command_interface: str
    u_cmd_topic: str
    goal_topic: str
    obs_topic: str
    reward_topic: str
    done_topic: str
    slack_topic: str
    reset_service: str
    success_window: int
    success_position_tolerance: float
    success_velocity_tolerance: float
    success_slack_accum: float
    publish_slack: bool
    use_cbf: bool
    max_episode_steps: int
    progress_reward_scale: float
    time_penalty: float
    slack_penalty: float
    infeasible_penalty: float
    collision_penalty: float
    joint_position_lower: List[float]
    joint_position_upper: List[float]
    joint_velocity_limits: List[float]
    initial_position_lower: List[float]
    initial_position_upper: List[float]
    random_seed: int

    @staticmethod
    def _as_list(value: Sequence[float], name: str, expected_len: int) -> List[float]:
        data = list(float(v) for v in value)
        if len(data) != expected_len:
            raise ValueError(f"Parameter '{name}' expected {expected_len} elements but received {len(data)}")
        return data

    @classmethod
    def from_node(cls, node: Node) -> "BridgeConfig":
        params = node.get_parameters(
            [
                "control_frequency",
                "joint_names",
                "command_topic",
                "command_interface",
                "u_cmd_topic",
                "goal_topic",
                "obs_topic",
                "reward_topic",
                "done_topic",
                "slack_topic",
                "reset_service",
                "success_window",
                "success_position_tolerance",
                "success_velocity_tolerance",
                "success_slack_accum",
                "publish_slack",
                "use_cbf",
                "max_episode_steps",
                "progress_reward_scale",
                "time_penalty",
                "slack_penalty",
                "infeasible_penalty",
                "collision_penalty",
                "joint_position_limits.lower",
                "joint_position_limits.upper",
                "joint_velocity_limits",
                "initial_position_range.lower",
                "initial_position_range.upper",
                "random_seed",
            ]
        )
        values: Dict[str, object] = {param.name: param.value for param in params}
        joint_names = list(values.get("joint_names", []))
        if not joint_names:
            raise ValueError("Parameter 'joint_names' must contain at least one joint name.")
        expected_len = len(joint_names)
        joint_lower = cls._as_list(values.get("joint_position_limits.lower", []), "joint_position_limits.lower", expected_len)
        joint_upper = cls._as_list(values.get("joint_position_limits.upper", []), "joint_position_limits.upper", expected_len)
        joint_velocity_limits = cls._as_list(values.get("joint_velocity_limits", []), "joint_velocity_limits", expected_len)
        initial_lower = cls._as_list(values.get("initial_position_range.lower", []), "initial_position_range.lower", expected_len)
        initial_upper = cls._as_list(values.get("initial_position_range.upper", []), "initial_position_range.upper", expected_len)
        random_seed = int(values.get("random_seed", 0) or 0)
        return cls(
            control_frequency=float(values.get("control_frequency", 50.0)),
            joint_names=joint_names,
            command_topic=str(values.get("command_topic", "/forward_position_controller/commands")),
            command_interface=str(values.get("command_interface", "position")),
            u_cmd_topic=str(values.get("u_cmd_topic", "/hrl/u_cmd")),
            goal_topic=str(values.get("goal_topic", "/hrl/goal")),
            obs_topic=str(values.get("obs_topic", "/hrl/obs")),
            reward_topic=str(values.get("reward_topic", "/hrl/reward")),
            done_topic=str(values.get("done_topic", "/hrl/done")),
            slack_topic=str(values.get("slack_topic", "/hrl/slack")),
            reset_service=str(values.get("reset_service", "/hrl/reset")),
            success_window=int(values.get("success_window", 5)),
            success_position_tolerance=float(values.get("success_position_tolerance", 0.0175)),
            success_velocity_tolerance=float(values.get("success_velocity_tolerance", 0.02)),
            success_slack_accum=float(values.get("success_slack_accum", 1e-3)),
            publish_slack=bool(values.get("publish_slack", True)),
            use_cbf=bool(values.get("use_cbf", False)),
            max_episode_steps=int(values.get("max_episode_steps", 500)),
            progress_reward_scale=float(values.get("progress_reward_scale", 1.0)),
            time_penalty=float(values.get("time_penalty", 0.02)),
            slack_penalty=float(values.get("slack_penalty", 5.0)),
            infeasible_penalty=float(values.get("infeasible_penalty", 5.0)),
            collision_penalty=float(values.get("collision_penalty", 5.0)),
            joint_position_lower=joint_lower,
            joint_position_upper=joint_upper,
            joint_velocity_limits=joint_velocity_limits,
            initial_position_lower=initial_lower,
            initial_position_upper=initial_upper,
            random_seed=random_seed,
        )
