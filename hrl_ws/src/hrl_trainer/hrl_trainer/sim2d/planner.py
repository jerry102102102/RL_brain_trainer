from __future__ import annotations

import numpy as np


class HighLevelHeuristicPlanner:
    """Heuristic strategic planner that mimics a frozen high-level LLM policy.

    It outputs a compact option/subgoal packet instead of low-level control.
    """

    def __init__(self, waypoint_scale: float = 0.35) -> None:
        self.waypoint_scale = waypoint_scale

    def plan(self, obs: np.ndarray) -> dict:
        x, y, yaw, v, omega, gx, gy = obs.tolist()
        dx, dy = gx - x, gy - y
        dist = float(np.hypot(dx, dy))

        # pseudo-option logic
        if dist > 0.8:
            option_id = "CRUISE"
            local_goal = np.array([x + self.waypoint_scale * dx, y + self.waypoint_scale * dy], dtype=np.float32)
        elif dist > 0.25:
            option_id = "APPROACH"
            local_goal = np.array([x + 0.6 * dx, y + 0.6 * dy], dtype=np.float32)
        else:
            option_id = "DOCK"
            local_goal = np.array([gx, gy], dtype=np.float32)

        return {
            "option_id": option_id,
            "subgoal_xy": local_goal,
            "termination": {"metric": "distance", "threshold": 0.08},
            "constraints": ["smooth_control", "bounded_turn_rate"],
        }
