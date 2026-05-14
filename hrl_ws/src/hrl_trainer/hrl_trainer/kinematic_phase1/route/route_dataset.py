"""Dense q-goal route dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..kinematics.fk_interface import compute_ee_pose6


@dataclass(frozen=True)
class RouteWaypoint:
    route_index: int
    q_goal: np.ndarray
    ee_target_pose6: np.ndarray
    next_q_delta: np.ndarray
    route_progress_m: float
    chunk_id: int


@dataclass(frozen=True)
class RouteDataset:
    path: Path
    waypoints: tuple[RouteWaypoint, ...]

    @property
    def q_goals(self) -> np.ndarray:
        return np.asarray([wp.q_goal for wp in self.waypoints], dtype=float)

    @property
    def poses6(self) -> np.ndarray:
        return np.asarray([wp.ee_target_pose6 for wp in self.waypoints], dtype=float)

    def __len__(self) -> int:
        return len(self.waypoints)

    def waypoint(self, route_index: int) -> RouteWaypoint:
        return self.waypoints[int(np.clip(route_index, 0, len(self.waypoints) - 1))]


def _extract_q(entry: object) -> np.ndarray:
    if isinstance(entry, dict):
        if "q" in entry:
            return np.asarray(entry["q"], dtype=float)
        if "q_goal" in entry:
            return np.asarray(entry["q_goal"], dtype=float)
    return np.asarray(entry, dtype=float)


def _chunk_id(route_index: int, chunk_bounds: Sequence[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(chunk_bounds):
        if lo <= route_index <= hi:
            return idx
    return len(chunk_bounds) - 1


def default_chunk_bounds(max_index: int) -> tuple[tuple[int, int], ...]:
    return (
        (1, min(40, max_index)),
        (41, min(80, max_index)),
        (81, min(120, max_index)),
        (121, min(180, max_index)),
        (181, min(260, max_index)),
        (261, min(360, max_index)),
        (361, max_index),
    )


def load_route_dataset(path: str | Path, *, chunk_bounds: Sequence[tuple[int, int]] | None = None) -> RouteDataset:
    route_path = Path(path)
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    entries = payload.get("route_q") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Route dataset must contain a non-empty list: {route_path}")

    q_goals = np.asarray([_extract_q(entry) for entry in entries], dtype=float)
    poses = np.asarray([compute_ee_pose6(q) for q in q_goals], dtype=float)
    pos_steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) if len(poses) > 1 else np.zeros(0)
    progress = np.concatenate([[0.0], np.cumsum(pos_steps)])
    bounds = tuple(chunk_bounds) if chunk_bounds is not None else default_chunk_bounds(len(q_goals) - 1)

    waypoints: list[RouteWaypoint] = []
    for idx, q in enumerate(q_goals):
        next_q_delta = q_goals[min(idx + 1, len(q_goals) - 1)] - q
        waypoints.append(
            RouteWaypoint(
                route_index=idx,
                q_goal=q.astype(float),
                ee_target_pose6=poses[idx].astype(float),
                next_q_delta=next_q_delta.astype(float),
                route_progress_m=float(progress[idx]),
                chunk_id=_chunk_id(idx, bounds),
            )
        )
    return RouteDataset(path=route_path, waypoints=tuple(waypoints))

