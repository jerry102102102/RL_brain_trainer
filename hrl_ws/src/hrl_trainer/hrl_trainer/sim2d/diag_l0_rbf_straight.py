from __future__ import annotations

import json

import numpy as np

from .env import DisturbanceConfig, Sim2DEnv
from .train_rl_brainer_v3_online import _rbf_controller


def run_diag_l0() -> dict:
    """Validate low-level RBF tracks a straight reference without any learning."""
    env = Sim2DEnv(seed=0, max_steps=120, level="easy", obstacle_count=0)
    env.disturbance = DisturbanceConfig(
        sensor_noise_std=0.0,
        sensor_bias_prob=0.0,
        sensor_bias_scale=0.0,
        action_delay_steps=0,
        friction_drag=0.08,
        impulse_prob=0.0,
        impulse_scale=0.0,
        obs_dropout_prob=0.0,
    )
    env.reset()
    env.obstacles = []
    env.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32)
    env.steps = 0

    desired = np.array([0.7, 0.0], dtype=np.float32)
    speed_err = []
    yaw_rate_err = []
    lateral = []

    for _ in range(100):
        obs = env._observe(env.state.copy())
        action = _rbf_controller(obs, desired)
        obs, _, done, _ = env.step(action)
        speed_err.append(float(desired[0] - obs[3]))
        yaw_rate_err.append(float(desired[1] - obs[4]))
        lateral.append(float(obs[1]))
        if done:
            break

    speed_rmse = float(np.sqrt(np.mean(np.square(speed_err))))
    yaw_rate_rmse = float(np.sqrt(np.mean(np.square(yaw_rate_err))))
    lateral_rmse = float(np.sqrt(np.mean(np.square(lateral))))
    final_x = float(env.state[0])
    final_y = float(env.state[1])

    pass_flag = bool(
        speed_rmse < 0.20
        and yaw_rate_rmse < 0.10
        and lateral_rmse < 0.05
        and abs(final_y) < 0.08
        and final_x > 1.0
    )

    return {
        "layer": "L0",
        "name": "RBF straight-line tracking (no learning)",
        "pass": pass_flag,
        "metrics": {
            "speed_rmse": speed_rmse,
            "yaw_rate_rmse": yaw_rate_rmse,
            "lateral_rmse": lateral_rmse,
            "final_x": final_x,
            "final_y": final_y,
            "steps_executed": len(speed_err),
        },
        "thresholds": {
            "speed_rmse_max": 0.20,
            "yaw_rate_rmse_max": 0.10,
            "lateral_rmse_max": 0.05,
            "abs_final_y_max": 0.08,
            "final_x_min": 1.0,
        },
    }


def main() -> None:
    print(json.dumps(run_diag_l0(), indent=2))


if __name__ == "__main__":
    main()
