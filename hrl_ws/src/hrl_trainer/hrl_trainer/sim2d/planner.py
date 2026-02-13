from __future__ import annotations

import numpy as np


class HighLevelHeuristicPlanner:
    """Heuristic strategic planner that mimics a frozen high-level LLM policy.

    It outputs a compact option/subgoal packet instead of low-level control.
    """

    def __init__(self, waypoint_scale: float = 0.35) -> None:
        self.waypoint_scale = waypoint_scale

    def plan(self, obs: np.ndarray) -> dict:
        base = obs[:7]
        x, y, yaw, v, omega, gx, gy = base.tolist()
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


class HighLevelHeuristicPlannerV2(HighLevelHeuristicPlanner):
    """Richer strategic layer for RL-brainer v2 with docking-specialized options."""

    def plan(self, obs: np.ndarray) -> dict:
        base = obs[:7]
        x, y, yaw, v, omega, gx, gy = base.tolist()
        dx, dy = gx - x, gy - y
        dist = float(np.hypot(dx, dy))
        target_heading = float(np.arctan2(dy, dx))
        heading_err = float((target_heading - yaw + np.pi) % (2 * np.pi) - np.pi)

        # Docking-specialized policy: split final phase into align and approach.
        if dist <= 0.35:
            if abs(heading_err) > 0.28:
                option_id = "DOCK_ALIGN"
                scale = 0.65
                speed_hint = 0.10
                constraints = ["stable_heading", "minimal_forward_speed"]
            else:
                option_id = "DOCK_APPROACH"
                scale = 1.0
                speed_hint = 0.18
                constraints = ["smooth_control", "soft_landing"]
        elif dist > 1.0:
            option_id = "CRUISE"
            scale = 0.40
            speed_hint = 0.90
            constraints = ["smooth_control"]
        elif abs(heading_err) > 0.7:
            option_id = "TURN_ALIGN"
            scale = 0.25
            speed_hint = 0.45
            constraints = ["bounded_turn_rate", "stable_heading"]
        else:
            option_id = "APPROACH"
            scale = 0.55
            speed_hint = 0.65
            constraints = ["smooth_control", "bounded_turn_rate"]

        local_goal = np.array([x + scale * dx, y + scale * dy], dtype=np.float32)
        return {
            "option_id": option_id,
            "subgoal_xy": local_goal,
            "speed_hint": speed_hint,
            "heading_err": heading_err,
            "termination": {"metric": "distance", "threshold": 0.08, "stable_steps": 4},
            "constraints": constraints,
        }
