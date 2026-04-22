"""Joint-limit helpers for the Phase 1 kinematic environment."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

JOINT_ORDER: tuple[str, ...] = (
    "Rack_joint",
    "robot_base_joint",
    "shoulder1_joint",
    "shoulder2_joint",
    "wr1_joint",
    "wr2_joint",
    "wr3_joint",
)


@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float
    upper: float
    delta_limit: float
    continuous: bool = False

    @property
    def span(self) -> float:
        return float(self.upper - self.lower)


def _default_specs() -> tuple[JointSpec, ...]:
    pi = math.pi
    return (
        JointSpec("Rack_joint", -0.385, 0.385, 0.08, continuous=False),
        JointSpec("robot_base_joint", -pi, pi, 0.30, continuous=False),
        JointSpec("shoulder1_joint", -pi, pi, 0.24, continuous=False),
        JointSpec("shoulder2_joint", -pi, pi, 0.24, continuous=False),
        JointSpec("wr1_joint", -pi, pi, 0.30, continuous=False),
        JointSpec("wr2_joint", -pi, pi, 0.40, continuous=True),
        JointSpec("wr3_joint", -pi, pi, 0.30, continuous=False),
    )


def _repo_root(start: Path) -> Path | None:
    for parent in start.parents:
        if (parent / "external" / "ENPM662_Group4_FinalProject").exists():
            return parent
    return None


def _urdf_path() -> Path | None:
    root = _repo_root(Path(__file__).resolve())
    if root is None:
        return None
    candidate = (
        root
        / "external"
        / "ENPM662_Group4_FinalProject"
        / "install"
        / "kitchen_robot_description"
        / "share"
        / "kitchen_robot_description"
        / "urdf"
        / "Kitchen_Robot_UR7DF.SLDASM.urdf"
    )
    return candidate if candidate.exists() else None


def _load_limits_from_urdf() -> dict[str, tuple[float, float]]:
    path = _urdf_path()
    if path is None:
        return {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {}

    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in JOINT_ORDER:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is None or upper is None:
            continue
        try:
            limits[name] = (float(lower), float(upper))
        except ValueError:
            continue
    return limits


def default_joint_specs() -> tuple[JointSpec, ...]:
    defaults = list(_default_specs())
    urdf_limits = _load_limits_from_urdf()
    resolved: list[JointSpec] = []
    for spec in defaults:
        lower, upper = urdf_limits.get(spec.name, (spec.lower, spec.upper))
        resolved.append(
            JointSpec(
                name=spec.name,
                lower=lower,
                upper=upper,
                delta_limit=spec.delta_limit,
                continuous=spec.continuous,
            )
        )
    return tuple(resolved)


def lower_bounds(specs: Iterable[JointSpec]) -> np.ndarray:
    return np.asarray([spec.lower for spec in specs], dtype=float)


def upper_bounds(specs: Iterable[JointSpec]) -> np.ndarray:
    return np.asarray([spec.upper for spec in specs], dtype=float)


def delta_limits(specs: Iterable[JointSpec]) -> np.ndarray:
    return np.asarray([spec.delta_limit for spec in specs], dtype=float)


def clip_joint_configuration(q: np.ndarray, specs: Iterable[JointSpec]) -> np.ndarray:
    specs_list = tuple(specs)
    return np.clip(np.asarray(q, dtype=float), lower_bounds(specs_list), upper_bounds(specs_list))


def sample_joint_configuration(
    rng: np.random.Generator,
    specs: Iterable[JointSpec],
    margin_fraction: float = 0.1,
) -> np.ndarray:
    specs_list = tuple(specs)
    lowers = lower_bounds(specs_list)
    uppers = upper_bounds(specs_list)
    spans = uppers - lowers
    margin = np.maximum(spans * margin_fraction, 1e-6)
    low = lowers + margin
    high = uppers - margin
    return rng.uniform(low=low, high=high, size=(len(specs_list),)).astype(float)


def normalize_joint_positions(q: np.ndarray, specs: Iterable[JointSpec]) -> np.ndarray:
    specs_list = tuple(specs)
    lowers = lower_bounds(specs_list)
    spans = upper_bounds(specs_list) - lowers
    normalized = 2.0 * ((np.asarray(q, dtype=float) - lowers) / np.maximum(spans, 1e-9)) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def normalize_joint_deltas(dq: np.ndarray, specs: Iterable[JointSpec]) -> np.ndarray:
    limits = np.maximum(delta_limits(specs), 1e-9)
    return np.clip(np.asarray(dq, dtype=float) / limits, -1.0, 1.0)


def joint_limit_margin(q: np.ndarray, specs: Iterable[JointSpec]) -> np.ndarray:
    specs_list = tuple(specs)
    lowers = lower_bounds(specs_list)
    uppers = upper_bounds(specs_list)
    q_arr = np.asarray(q, dtype=float)
    left = (q_arr - lowers) / np.maximum(uppers - lowers, 1e-9)
    right = (uppers - q_arr) / np.maximum(uppers - lowers, 1e-9)
    margin = 2.0 * np.minimum(left, right)
    return np.clip(margin, 0.0, 1.0)
