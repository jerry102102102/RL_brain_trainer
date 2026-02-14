from __future__ import annotations

import json

import numpy as np

from .planner import HighLevelHeuristicPlannerV2


def run_diag_l1(num_samples: int = 256) -> dict:
    """Validate high-level planner emits straight-line-consistent subgoals."""
    rng = np.random.default_rng(0)
    planner = HighLevelHeuristicPlannerV2(waypoint_scale=0.35)

    orth_errors = []
    ratios = []
    invalid = 0

    for _ in range(num_samples):
        x, y = rng.uniform(-1.5, 1.5, size=2)
        gx, gy = rng.uniform(-1.5, 1.5, size=2)
        if np.hypot(gx - x, gy - y) < 1e-3:
            gx += 0.1
        yaw = rng.uniform(-np.pi, np.pi)
        v = rng.uniform(-0.1, 0.8)
        omega = rng.uniform(-0.4, 0.4)

        obs = np.array([x, y, yaw, v, omega, gx, gy, 0.0, 0.0, 0.0], dtype=np.float32)
        packet = planner.plan(obs)
        subgoal = np.asarray(packet["subgoal_xy"], dtype=np.float64)

        goal_vec = np.array([gx - x, gy - y], dtype=np.float64)
        sub_vec = np.array([subgoal[0] - x, subgoal[1] - y], dtype=np.float64)
        goal_norm = float(np.linalg.norm(goal_vec))
        if goal_norm < 1e-6:
            invalid += 1
            continue

        # Signed area in 2D normalized by goal magnitude => orthogonal deviation.
        orth = float(abs(goal_vec[0] * sub_vec[1] - goal_vec[1] * sub_vec[0]) / goal_norm)
        ratio = float(np.dot(sub_vec, goal_vec) / (goal_norm * goal_norm))
        orth_errors.append(orth)
        ratios.append(ratio)

    max_orth = float(np.max(orth_errors)) if orth_errors else float("inf")
    min_ratio = float(np.min(ratios)) if ratios else -float("inf")
    max_ratio = float(np.max(ratios)) if ratios else float("inf")

    pass_flag = bool(
        invalid == 0
        and max_orth < 1e-6
        and min_ratio > 0.0
        and max_ratio <= 1.01
    )

    return {
        "layer": "L1",
        "name": "Planner straight-line-consistent subgoals (obstacle-free)",
        "pass": pass_flag,
        "metrics": {
            "samples": int(num_samples),
            "invalid_samples": int(invalid),
            "max_orthogonal_error": max_orth,
            "min_progress_ratio": min_ratio,
            "max_progress_ratio": max_ratio,
        },
        "thresholds": {
            "invalid_samples_eq": 0,
            "max_orthogonal_error_max": 1e-6,
            "min_progress_ratio_min": 0.0,
            "max_progress_ratio_max": 1.01,
        },
    }


def main() -> None:
    print(json.dumps(run_diag_l1(), indent=2))


if __name__ == "__main__":
    main()
