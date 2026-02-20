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
    command_noise_std_v: float = 0.0
    command_noise_std_omega: float = 0.0
    gust_prob: float = 0.0
    gust_scale_v: float = 0.0
    gust_scale_omega: float = 0.0
    gust_cooldown_steps: int = 0


class Sim2DEnv:
    """Simple 2D dynamics env with acceleration + steering, disturbances and obstacles.

    base_state = [x, y, yaw, v, omega, goal_x, goal_y]
    obs = base_state + [nearest_obs_dx, nearest_obs_dy, nearest_obs_dist]
    action = [a_linear, a_angular] in [-1, 1]
    """

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 250,
        level: str = "easy",
        obstacle_count: int = 3,
        control_mode: str = "velocity",
        min_start_goal_dist: float = 1.1,
        min_obstacle_spacing: float = 0.22,
        corridor_clearance: float = 0.14,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.dt = 0.1
        self._delay_buffer: list[np.ndarray] = []
        self.level = level
        self.control_mode = str(control_mode)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.min_obstacle_spacing = float(min_obstacle_spacing)
        self.corridor_clearance = float(corridor_clearance)
        self.disturbance = self._build_disturbance(level)
        # Collision checks approximate a regular pentagon footprint by its circumscribed radius.
        self.robot_sides = 5
        self.robot_apothem = 0.09
        self.robot_circ_radius = float(self.robot_apothem / math.cos(math.pi / self.robot_sides))
        self.world_half_extent = 1.6
        self.state = np.zeros(7, dtype=np.float32)
        self.steps = 0
        self.obstacle_count = int(obstacle_count)
        self.obstacles: list[tuple[float, float, float]] = []
        self._gust_cooldown = 0

    def _build_disturbance(self, level: str) -> DisturbanceConfig:
        if level == "easy":
            return DisturbanceConfig(
                sensor_noise_std=0.005,
                friction_drag=0.08,
                impulse_prob=0.003,
                command_noise_std_v=0.01,
                command_noise_std_omega=0.02,
            )
        if level == "medium":
            return DisturbanceConfig(
                sensor_noise_std=0.01,
                action_delay_steps=1,
                friction_drag=0.12,
                impulse_prob=0.015,
                command_noise_std_v=0.02,
                command_noise_std_omega=0.04,
                gust_prob=0.01,
                gust_scale_v=0.08,
                gust_scale_omega=0.12,
                gust_cooldown_steps=12,
            )
        if level == "hard":
            return DisturbanceConfig(
                sensor_noise_std=0.02,
                sensor_bias_prob=0.05,
                sensor_bias_scale=0.12,
                action_delay_steps=2,
                friction_drag=0.2,
                impulse_prob=0.03,
                impulse_scale=0.25,
                obs_dropout_prob=0.04,
                command_noise_std_v=0.03,
                command_noise_std_omega=0.07,
                gust_prob=0.02,
                gust_scale_v=0.12,
                gust_scale_omega=0.2,
                gust_cooldown_steps=14,
            )
        return DisturbanceConfig()

    def reset(self) -> np.ndarray:
        x, y = self.rng.uniform(-1.0, 1.0, size=2)
        yaw = self.rng.uniform(-math.pi, math.pi)
        gx, gy = self.rng.uniform(-1.5, 1.5, size=2)
        attempts = 0
        while math.hypot(gx - x, gy - y) < self.min_start_goal_dist and attempts < 64:
            gx, gy = self.rng.uniform(-1.5, 1.5, size=2)
            attempts += 1

        self.state = np.array([x, y, yaw, 0.0, 0.0, gx, gy], dtype=np.float32)
        self.steps = 0
        self._delay_buffer.clear()
        self._gust_cooldown = 0
        self.obstacles = self._sample_obstacles((x, y), (gx, gy))
        return self._observe(self.state.copy())

    def step(self, action: np.ndarray):
        self.steps += 1

        x, y, yaw, v, omega, gx, gy = self.state

        if self.control_mode == "velocity":
            # command is [v_cmd, omega_cmd]
            # v_cmd now supports reverse motion via symmetric absolute bound.
            # omega_cmd is left unconstrained to allow free turning.
            action = np.asarray(action, dtype=np.float32)
            action = np.array([np.clip(action[0], -1.2, 1.2), float(action[1])], dtype=np.float32)
        else:
            # legacy acceleration mode
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.disturbance.action_delay_steps > 0:
            self._delay_buffer.append(action)
            if len(self._delay_buffer) <= self.disturbance.action_delay_steps:
                action = np.zeros_like(action)
            else:
                action = self._delay_buffer.pop(0)

        if self.control_mode == "velocity":
            cmd_v = float(action[0] + self.rng.normal(0.0, self.disturbance.command_noise_std_v))
            cmd_o = float(action[1] + self.rng.normal(0.0, self.disturbance.command_noise_std_omega))
            cmd_v = float(np.clip(cmd_v, -1.2, 1.2))

            if self._gust_cooldown > 0:
                self._gust_cooldown -= 1
            elif self.rng.random() < self.disturbance.gust_prob:
                cmd_v += float(self.rng.normal(0.0, self.disturbance.gust_scale_v))
                cmd_o += float(self.rng.normal(0.0, self.disturbance.gust_scale_omega))
                self._gust_cooldown = int(self.disturbance.gust_cooldown_steps)

            cmd_v = float(np.clip(cmd_v, -1.2, 1.2))

            # first-order velocity tracking + drag
            alpha_v = 0.38
            alpha_o = 0.42
            v = (1.0 - alpha_v) * v + alpha_v * cmd_v
            omega = (1.0 - alpha_o) * omega + alpha_o * cmd_o
            v *= (1.0 - self.disturbance.friction_drag * 0.25 * self.dt)
            omega *= (1.0 - self.disturbance.friction_drag * 0.2 * self.dt)
        else:
            a_lin, a_ang = action
            # disturbance impulse
            if self.rng.random() < self.disturbance.impulse_prob:
                v += self.rng.normal(0, self.disturbance.impulse_scale)
                omega += self.rng.normal(0, self.disturbance.impulse_scale * 0.5)

            # legacy accel dynamics with drag/friction
            v = (1.0 - self.disturbance.friction_drag * self.dt) * v + a_lin * self.dt
            omega = (1.0 - self.disturbance.friction_drag * 0.5 * self.dt) * omega + a_ang * self.dt

        yaw = yaw + omega * self.dt
        x = x + v * math.cos(yaw) * self.dt
        y = y + v * math.sin(yaw) * self.dt

        self.state = np.array([x, y, yaw, v, omega, gx, gy], dtype=np.float32)

        dist = float(np.linalg.norm(np.array([gx - x, gy - y], dtype=np.float32)))

        obstacle_contact = False
        for ox, oy, rr in self.obstacles:
            contact_r = float(rr + self.robot_circ_radius)
            if (x - ox) ** 2 + (y - oy) ** 2 <= contact_r ** 2:
                obstacle_contact = True
                break

        wall_contact = bool(
            abs(float(x)) >= (self.world_half_extent - self.robot_circ_radius)
            or abs(float(y)) >= (self.world_half_extent - self.robot_circ_radius)
        )
        collided = bool(obstacle_contact or wall_contact)
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
            "obstacle_contact": bool(obstacle_contact),
            "wall_contact": bool(wall_contact),
            "control_effort": float(np.linalg.norm(action)),
        }
        return obs, reward, done, info

    def _point_to_segment_distance(self, px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-9:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    def _sample_obstacles(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float, float]]:
        obs = []
        sx, sy = start_xy
        gx, gy = goal_xy
        attempts = 0
        while len(obs) < self.obstacle_count and attempts < 240:
            attempts += 1
            pad = self.robot_circ_radius + 0.05
            ox, oy = self.rng.uniform(-(self.world_half_extent - pad), (self.world_half_extent - pad), size=2)
            rr = float(self.rng.uniform(0.12, 0.22))

            # keep start/goal launch zones clear
            if (ox - sx) ** 2 + (oy - sy) ** 2 < (rr + 0.28) ** 2:
                continue
            if (ox - gx) ** 2 + (oy - gy) ** 2 < (rr + 0.28) ** 2:
                continue

            # avoid placing obstacles right on the direct corridor to reduce random deadlocks
            d_seg = self._point_to_segment_distance(float(ox), float(oy), float(sx), float(sy), float(gx), float(gy))
            if d_seg < (self.corridor_clearance + rr):
                continue

            # spread obstacles to reduce pathological clusters
            ok = True
            for ex, ey, er in obs:
                if (ox - ex) ** 2 + (oy - ey) ** 2 < (rr + er + self.min_obstacle_spacing) ** 2:
                    ok = False
                    break
            if not ok:
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
            d = math.hypot(dx, dy) - rr - self.robot_circ_radius
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
