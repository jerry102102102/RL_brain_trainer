"""Gymnasium-compatible route curriculum environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv, Phase1EnvConfig
from ..envs.spaces import EnvBase
from ..kinematics.fk_interface import compute_ee_pose6
from .reward_route import RouteRewardConfig, compute_route_reward, route_ready
from .route_dataset import RouteDataset
from .route_observation import RouteObservationConfig, augment_route_observation, build_route_observation_space
from .route_reset_samplers import RouteResetSamplerConfig, sample_route_reset


@dataclass(frozen=True)
class RouteEnvConfig:
    base_env_config: Phase1EnvConfig
    reset_config: RouteResetSamplerConfig
    reward_config: RouteRewardConfig
    observation_config: RouteObservationConfig = RouteObservationConfig()


class RouteKinematicEnv(EnvBase):
    """A light wrapper that reuses ArmKinematicEnv with route-specific resets/reward."""

    metadata = {"render_modes": []}

    def __init__(self, *, route: RouteDataset, config: RouteEnvConfig, seed: int | None = None) -> None:
        self.route = route
        self.config = config
        self.base_env = ArmKinematicEnv(config=config.base_env_config)
        self.action_space = self.base_env.action_space
        self.observation_space = (
            build_route_observation_space(self.base_env.observation_space, config.base_env_config.n_joints)
            if config.observation_config.include_route_keys
            else self.base_env.observation_space
        )
        self._rng = np.random.default_rng(seed)
        self._route_index = 1
        self._start_route_index = 0
        self._ready_streak = 0
        self._prev_info: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        if "route_index" in opts:
            route_index = int(opts["route_index"])
            start_index = int(opts.get("start_route_index", max(route_index - 1, 0)))
            initial_q = self.route.waypoint(start_index).q_goal
            goal_q = self.route.waypoint(route_index).q_goal
            initial_dq = np.zeros_like(initial_q)
            initial_prev_action = np.zeros_like(initial_q)
            reset_mode = "explicit"
        else:
            sample = sample_route_reset(
                rng=self._rng,
                route=self.route,
                joint_specs=self.config.base_env_config.joint_specs,
                config=self.config.reset_config,
            )
            route_index = sample.route_index
            start_index = sample.start_route_index
            initial_q = sample.initial_q
            goal_q = sample.goal_q
            initial_dq = sample.initial_dq
            initial_prev_action = sample.initial_prev_action
            reset_mode = sample.reset_mode
        self._route_index = int(route_index)
        self._start_route_index = int(start_index)
        self._ready_streak = 0
        obs, info = self.base_env.reset(
            seed=seed,
            options={
                "initial_q": initial_q,
                "initial_dq": initial_dq,
                "initial_prev_action": initial_prev_action,
                "goal_q": goal_q,
                "policy_mode": "approach",
            },
        )
        info.update(
            {
                "route_index": self._route_index,
                "start_route_index": self._start_route_index,
                "route_reset_mode": reset_mode,
                "route_progress_m": self.route.waypoint(self._route_index).route_progress_m,
                "route_chunk_id": self.route.waypoint(self._route_index).chunk_id,
            }
        )
        self._prev_info = dict(info)
        return self._augment_obs(obs), info

    def set_route_window(self, *, max_route_index: int, min_route_index: int = 1) -> None:
        reset_cfg = self.config.reset_config
        self.config = RouteEnvConfig(
            base_env_config=self.config.base_env_config,
            reward_config=self.config.reward_config,
            observation_config=self.config.observation_config,
            reset_config=RouteResetSamplerConfig(
                mode=reset_cfg.mode,
                min_route_index=int(min_route_index),
                max_route_index=int(max_route_index),
                segment_start_index=reset_cfg.segment_start_index,
                segment_end_index=reset_cfg.segment_end_index,
                replay_start_index=reset_cfg.replay_start_index,
                replay_end_index=reset_cfg.replay_end_index,
                prefix_start_reset_ratio=reset_cfg.prefix_start_reset_ratio,
                random_prefix_reset_ratio=reset_cfg.random_prefix_reset_ratio,
                segment_reset_ratio=reset_cfg.segment_reset_ratio,
                replay_reset_ratio=reset_cfg.replay_reset_ratio,
                recovery_reset_ratio=reset_cfg.recovery_reset_ratio,
                q_noise_std=reset_cfg.q_noise_std,
                dq_noise_std=reset_cfg.dq_noise_std,
                prev_action_noise_std=reset_cfg.prev_action_noise_std,
            ),
        )

    def step(self, action):
        prev_q = np.asarray(self._prev_info["q"], dtype=float) if self._prev_info is not None else self.base_env._q.copy()  # noqa: SLF001
        prev_dq = np.asarray(self._prev_info["dq"], dtype=float) if self._prev_info is not None else self.base_env._dq.copy()  # noqa: SLF001
        prev_pose6 = compute_ee_pose6(prev_q)
        prev_action = self.base_env._prev_action.copy()  # noqa: SLF001
        obs, _, base_terminated, truncated, info = self.base_env.step(action)
        curr_q = np.asarray(info["q"], dtype=float)
        curr_dq = np.asarray(info["dq"], dtype=float)
        goal_q = self.route.waypoint(self._route_index).q_goal
        goal_pose6 = self.route.waypoint(self._route_index).ee_target_pose6
        tangent = self.route.waypoint(max(self._route_index - 1, 0)).next_q_delta
        nearest_q_dist = float(np.min(np.linalg.norm(self.route.q_goals - curr_q, axis=1)))

        curr_pose6 = compute_ee_pose6(curr_q)
        action_arr = np.asarray(action, dtype=float)
        q_error_norm = float(np.linalg.norm(goal_q - curr_q))
        prev_q_error_norm = float(np.linalg.norm(goal_q - prev_q))
        action_norm = float(np.linalg.norm(action_arr))
        dq_norm = float(np.linalg.norm(curr_dq))
        ready_now = route_ready(
            q_error_norm=q_error_norm,
            pos_error_norm=float(info["position_error_norm"]),
            ori_error_norm=float(info["orientation_error_norm"]),
            action_norm=action_norm,
            dq_norm=dq_norm,
            config=self.config.reward_config,
        )
        self._ready_streak = self._ready_streak + 1 if ready_now else 0
        reward, components = compute_route_reward(
            prev_q=prev_q,
            curr_q=curr_q,
            goal_q=goal_q,
            prev_pose6=prev_pose6,
            curr_pose6=curr_pose6,
            goal_pose6=goal_pose6,
            route_tangent_q=tangent,
            action=action_arr,
            prev_action=prev_action,
            prev_dq=prev_dq,
            curr_dq=curr_dq,
            ready_streak=self._ready_streak,
            nearest_route_q_distance=nearest_q_dist,
            config=self.config.reward_config,
        )
        success = bool(ready_now and self._ready_streak >= self.config.base_env_config.termination_config.success_dwell_steps)
        terminated = bool(base_terminated)
        if bool(base_terminated) and str(info.get("reason", "")) == "success" and not success:
            terminated = False
        if success and self.config.base_env_config.termination_config.terminate_on_success:
            terminated = True
            info["termination_reason"] = "route_ready_success"
        info.update(
            {
                "success": success,
                "route_ready": ready_now,
                "route_ready_streak": int(self._ready_streak),
                "route_q_error_norm": q_error_norm,
                "route_orientation_hit": bool(float(info["orientation_error_norm"]) <= self.config.reward_config.route_ready_ori_threshold_rad),
                "route_regression": bool(q_error_norm > prev_q_error_norm),
                "nearest_route_q_distance": nearest_q_dist,
                "route_index": self._route_index,
                "start_route_index": self._start_route_index,
                "route_progress_m": self.route.waypoint(self._route_index).route_progress_m,
                "route_chunk_id": self.route.waypoint(self._route_index).chunk_id,
                "reward_components": components,
            }
        )
        self._prev_info = dict(info)
        return self._augment_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def _augment_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        target = self.route.waypoint(self._route_index)
        return augment_route_observation(
            obs,
            q=self.base_env._q,  # noqa: SLF001
            q_goal=target.q_goal,
            route_tangent_q=self.route.waypoint(max(self._route_index - 1, 0)).next_q_delta,
            route_index=self._route_index,
            max_route_index=len(self.route) - 1,
            route_progress_m=target.route_progress_m,
            total_route_progress_m=self.route.waypoint(len(self.route) - 1).route_progress_m,
            joint_specs=self.config.base_env_config.joint_specs,
            enabled=self.config.observation_config.include_route_keys,
        )

    def close(self) -> None:
        close_fn = getattr(self.base_env, "close", None)
        if callable(close_fn):
            close_fn()
