from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class DisturbanceConfig:
    sensor_noise_std: float = 0.01
    sensor_bias_prob: float = 0.02
    sensor_bias_scale: float = 0.08
    action_delay_steps: int = 0
    friction_drag: float = 0.12
    impulse_prob: float = 0.02
    impulse_scale: float = 0.25
    obs_dropout_prob: float = 0.0


class Sim2DEnv:
    """Simple 2D dynamics env with acceleration + steering, disturbances and obstacles.

    base_state = [x, y, yaw, v, omega, goal_x, goal_y]
    obs = base_state + [nearest_obs_dx, nearest_obs_dy, nearest_obs_dist]
    action = [a_linear, a_angular] in [-1, 1]
    """

    def __init__(self, seed: int = 0, max_steps: int = 250, level: str = "easy", obstacle_count: int = 3) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.dt = 0.1
        self._delay_buffer: list[np.ndarray] = []
        self.level = level
        self.disturbance = self._build_disturbance(level)
        self.state = np.zeros(7, dtype=np.float32)
        self.steps = 0
        self.obstacle_count = int(obstacle_count)
        self.obstacles: list[tuple[float, float, float]] = []

    def _build_disturbance(self, level: str) -> DisturbanceConfig:
        if level == "easy":
            return DisturbanceConfig(sensor_noise_std=0.005, friction_drag=0.08, impulse_prob=0.005)
        if level == "medium":
            return DisturbanceConfig(sensor_noise_std=0.01, action_delay_steps=1, friction_drag=0.12, impulse_prob=0.02)
        if level == "hard":
            return DisturbanceConfig(
                sensor_noise_std=0.02,
                sensor_bias_prob=0.05,
                sensor_bias_scale=0.12,
                action_delay_steps=2,
                friction_drag=0.2,
                impulse_prob=0.05,
                impulse_scale=0.35,
                obs_dropout_prob=0.04,
            )
        return DisturbanceConfig()

    def reset(self) -> np.ndarray:
        x, y = self.rng.uniform(-1.0, 1.0, size=2)
        yaw = self.rng.uniform(-math.pi, math.pi)
        gx, gy = self.rng.uniform(-1.5, 1.5, size=2)
        self.state = np.array([x, y, yaw, 0.0, 0.0, gx, gy], dtype=np.float32)
        self.steps = 0
        self._delay_buffer.clear()
        self.obstacles = self._sample_obstacles((x, y), (gx, gy))
        return self._observe(self.state.copy())

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.disturbance.action_delay_steps > 0:
            self._delay_buffer.append(action)
            if len(self._delay_buffer) <= self.disturbance.action_delay_steps:
                action = np.zeros_like(action)
            else:
                action = self._delay_buffer.pop(0)

        x, y, yaw, v, omega, gx, gy = self.state
        a_lin, a_ang = action

        # disturbance impulse
        if self.rng.random() < self.disturbance.impulse_prob:
            v += self.rng.normal(0, self.disturbance.impulse_scale)
            omega += self.rng.normal(0, self.disturbance.impulse_scale * 0.5)

        # dynamics with drag/friction
        v = (1.0 - self.disturbance.friction_drag * self.dt) * v + a_lin * self.dt
        omega = (1.0 - self.disturbance.friction_drag * 0.5 * self.dt) * omega + a_ang * self.dt

        yaw = yaw + omega * self.dt
        x = x + v * math.cos(yaw) * self.dt
        y = y + v * math.sin(yaw) * self.dt

        self.state = np.array([x, y, yaw, v, omega, gx, gy], dtype=np.float32)

        dist = float(np.linalg.norm(np.array([gx - x, gy - y], dtype=np.float32)))

        collided = False
        for ox, oy, rr in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= rr ** 2:
                collided = True
                break

        done = bool(dist < 0.08 or self.steps >= self.max_steps or collided)
        success = bool(dist < 0.08 and not collided)

        control_penalty = 0.02 * float(np.linalg.norm(action))
        reward = -dist - control_penalty
        if success:
            reward += 5.0
        if collided:
            reward -= 6.0

        obs = self._observe(self.state.copy())
        info = {
            "distance": dist,
            "success": success,
            "collided": collided,
            "control_effort": float(np.linalg.norm(action)),
        }
        return obs, reward, done, info

    def _sample_obstacles(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float, float]]:
        obs = []
        sx, sy = start_xy
        gx, gy = goal_xy
        attempts = 0
        while len(obs) < self.obstacle_count and attempts < 100:
            attempts += 1
            ox, oy = self.rng.uniform(-1.4, 1.4, size=2)
            rr = float(self.rng.uniform(0.12, 0.22))
            if (ox - sx) ** 2 + (oy - sy) ** 2 < (rr + 0.25) ** 2:
                continue
            if (ox - gx) ** 2 + (oy - gy) ** 2 < (rr + 0.25) ** 2:
                continue
            obs.append((float(ox), float(oy), rr))
        return obs

    def _nearest_obstacle_feature(self, x: float, y: float) -> np.ndarray:
        if not self.obstacles:
            return np.zeros(3, dtype=np.float32)
        best = None
        best_d = 1e9
        for ox, oy, rr in self.obstacles:
            dx, dy = ox - x, oy - y
            d = math.hypot(dx, dy) - rr
            if d < best_d:
                best_d = d
                best = (dx, dy, d)
        assert best is not None
        return np.array(best, dtype=np.float32)

    def _observe(self, s: np.ndarray) -> np.ndarray:
        obs = s.copy()
        # gaussian sensor noise
        obs[:5] += self.rng.normal(0, self.disturbance.sensor_noise_std, size=5)

        # occasional sensor bias
        if self.rng.random() < self.disturbance.sensor_bias_prob:
            obs[:2] += self.rng.normal(0, self.disturbance.sensor_bias_scale, size=2)

        # random dropout
        if self.rng.random() < self.disturbance.obs_dropout_prob:
            obs[:5] = 0.0

        obstacle_feat = self._nearest_obstacle_feature(float(obs[0]), float(obs[1]))
        obs_full = np.concatenate([obs, obstacle_feat], dtype=np.float32)
        return obs_full.astype(np.float32)
