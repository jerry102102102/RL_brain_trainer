from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
import yaml

from .env import Sim2DEnv
from .planner import HighLevelHeuristicPlanner


def _heuristic_low_level(obs: np.ndarray, subgoal_xy: np.ndarray) -> np.ndarray:
    x, y, yaw, v, omega = obs[:5]
    dx, dy = float(subgoal_xy[0] - x), float(subgoal_xy[1] - y)
    desired_heading = float(np.arctan2(dy, dx))
    heading_error = (desired_heading - yaw + np.pi) % (2 * np.pi) - np.pi
    dist = float(np.hypot(dx, dy))

    a_lin = np.clip(1.2 * dist - 0.4 * v, -1.0, 1.0)
    a_ang = np.clip(1.8 * heading_error - 0.3 * omega, -1.0, 1.0)
    return np.array([a_lin, a_ang], dtype=np.float32)


def run_eval(cfg: dict) -> dict:
    episodes = int(cfg.get("episodes", 30))
    seed = int(cfg.get("seed", 0))
    level = str(cfg.get("disturbance_level", "easy"))
    max_steps = int(cfg.get("max_steps", 250))

    planner = HighLevelHeuristicPlanner(waypoint_scale=float(cfg.get("waypoint_scale", 0.35)))

    success = 0
    convergence_steps = []
    rmse_list = []
    recovery_times = []
    effort_list = []

    for ep in range(episodes):
        env = Sim2DEnv(seed=seed + ep, max_steps=max_steps, level=level)
        obs = env.reset()

        dist_hist = []
        effort = 0.0
        best_recovery = None
        disturbed = False
        disturbed_step = None

        for t in range(max_steps):
            packet = planner.plan(obs)
            action = _heuristic_low_level(obs, packet["subgoal_xy"])
            next_obs, reward, done, info = env.step(action)

            d = float(info["distance"])
            dist_hist.append(d)
            effort += float(info["control_effort"])

            if d > 1.0 and not disturbed:
                disturbed = True
                disturbed_step = t

            if disturbed and best_recovery is None and d < 0.3 and disturbed_step is not None:
                best_recovery = t - disturbed_step

            obs = next_obs
            if done:
                if info.get("success", False):
                    success += 1
                    convergence_steps.append(t + 1)
                break

        rmse = float(np.sqrt(np.mean(np.square(dist_hist)))) if dist_hist else float("nan")
        rmse_list.append(rmse)
        effort_list.append(effort)
        if best_recovery is not None:
            recovery_times.append(best_recovery)

    out = {
        "episodes": episodes,
        "success_rate": success / max(episodes, 1),
        "time_to_convergence": float(np.mean(convergence_steps)) if convergence_steps else None,
        "tracking_rmse": float(np.mean(rmse_list)) if rmse_list else None,
        "recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
        "control_effort": float(np.mean(effort_list)) if effort_list else None,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D v2 experiment smoke/eval")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", default="", help="Output JSON path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics = run_eval(cfg)
    print(json.dumps(metrics, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
