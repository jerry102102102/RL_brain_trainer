"""Phase 1 kinematic RL stack.

This package provides a pure forward-kinematics Gymnasium environment,
simple reward/termination logic, deterministic evaluation helpers, and
Stable-Baselines3 training entry points for rapid iteration.
"""

from .envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from .envs.reward_fn import RewardConfig
from .envs.termination import TerminationConfig

__all__ = [
    "ArmKinematicEnv",
    "Phase1EnvConfig",
    "RewardConfig",
    "TerminationConfig",
]
