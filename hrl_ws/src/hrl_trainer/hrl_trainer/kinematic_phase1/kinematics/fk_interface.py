"""Forward-kinematics interface for the Phase 1 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hrl_trainer.v5_1.ee_fk import ee_pose6_from_q

from .joint_limits import JointSpec, sample_joint_configuration


@dataclass(frozen=True)
class KinematicTarget:
    q: np.ndarray
    pose6: np.ndarray


def compute_ee_pose6(q: Sequence[float]) -> np.ndarray:
    return ee_pose6_from_q(np.asarray(q, dtype=float))


def sample_reachable_target(
    rng: np.random.Generator,
    joint_specs: Sequence[JointSpec],
    margin_fraction: float = 0.1,
) -> KinematicTarget:
    q_goal = sample_joint_configuration(rng, joint_specs, margin_fraction=margin_fraction)
    pose6 = compute_ee_pose6(q_goal)
    return KinematicTarget(q=q_goal, pose6=pose6)
