"""Sequential mini-route environment for dense q-goal curriculum training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..envs.arm_kinematic_env import ArmKinematicEnv
from ..envs.spaces import EnvBase
from ..kinematics.fk_interface import compute_ee_pose6
from .reward_route import compute_route_reward, route_ready
from .route_dataset import RouteDataset
from .route_env import RouteEnvConfig
from .route_observation import augment_route_observation, build_route_observation_space
from .route_reset_samplers import sample_route_reset


@dataclass(frozen=True)
class RouteSequenceConfig:
    """Controls how many consecutive waypoints one episode must consume."""

    enabled: bool = False
    sequence_length: int = 5
    reset_ready_streak_on_advance: bool = True


class RouteSequenceKinematicEnv(EnvBase):
    """Train on actual final-q chaining instead of independent waypoint episodes.

    The observation/action contract is intentionally identical to ArmKinematicEnv.
    The only change is episode structure: when a waypoint is reached, the same
    physical state immediately receives the next dense route waypoint as target.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        route: RouteDataset,
        config: RouteEnvConfig,
        sequence_config: RouteSequenceConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.route = route
        self.config = config
        self.sequence_config = sequence_config or RouteSequenceConfig()
        self.base_env = ArmKinematicEnv(config=config.base_env_config)
        self.action_space = self.base_env.action_space
        self.observation_space = (
            build_route_observation_space(self.base_env.observation_space, config.base_env_config.n_joints)
            if config.observation_config.include_route_keys
            else self.base_env.observation_space
        )
        self._rng = np.random.default_rng(seed)
        self._current_route_index = 1
        self._start_route_index = 0
        self._last_route_index = 1
        self._ready_streak = 0
        self._completed_waypoints = 0
        self._prev_info: dict[str, Any] | None = None

    def set_route_window(self, *, max_route_index: int, min_route_index: int = 1) -> None:
        from .route_env import RouteEnvConfig as _RouteEnvConfig
        from .route_reset_samplers import RouteResetSamplerConfig

        reset_cfg = self.config.reset_config
        self.config = _RouteEnvConfig(
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        if "route_index" in opts:
            first_target = int(opts["route_index"])
            start_index = int(opts.get("start_route_index", max(first_target - 1, 0)))
            initial_q = np.asarray(opts.get("initial_q", self.route.waypoint(start_index).q_goal), dtype=float)
            initial_dq = np.asarray(opts.get("initial_dq", np.zeros_like(initial_q)), dtype=float)
            initial_prev_action = np.asarray(opts.get("initial_prev_action", np.zeros_like(initial_q)), dtype=float)
            reset_mode = "explicit_sequence"
        else:
            sample = sample_route_reset(
                rng=self._rng,
                route=self.route,
                joint_specs=self.config.base_env_config.joint_specs,
                config=self.config.reset_config,
            )
            first_target = sample.route_index
            start_index = sample.start_route_index
            initial_q = sample.initial_q
            initial_dq = sample.initial_dq
            initial_prev_action = sample.initial_prev_action
            reset_mode = sample.reset_mode

        max_index = min(self.config.reset_config.max_route_index, len(self.route) - 1)
        sequence_len = max(int(self.sequence_config.sequence_length), 1)
        self._current_route_index = int(np.clip(first_target, 1, max_index))
        self._start_route_index = int(start_index)
        self._last_route_index = int(min(max_index, self._current_route_index + sequence_len - 1))
        self._ready_streak = 0
        self._completed_waypoints = 0

        obs, info = self.base_env.reset(
            seed=seed,
            options={
                "initial_q": initial_q,
                "initial_dq": initial_dq,
                "initial_prev_action": initial_prev_action,
                "goal_q": self.route.waypoint(self._current_route_index).q_goal,
                "policy_mode": "approach",
            },
        )
        info.update(self._route_info(reset_mode=reset_mode, waypoint_success=False, sequence_success=False))
        self._prev_info = dict(info)
        return self._augment_obs(obs), info

    def step(self, action):
        prev_q = np.asarray(self._prev_info["q"], dtype=float) if self._prev_info is not None else self.base_env._q.copy()  # noqa: SLF001
        prev_dq = np.asarray(self._prev_info["dq"], dtype=float) if self._prev_info is not None else self.base_env._dq.copy()  # noqa: SLF001
        prev_pose6 = compute_ee_pose6(prev_q)
        prev_action = self.base_env._prev_action.copy()  # noqa: SLF001
        target_index = self._current_route_index
        goal_q = self.route.waypoint(target_index).q_goal
        goal_pose6 = self.route.waypoint(target_index).ee_target_pose6
        tangent = self.route.waypoint(max(target_index - 1, 0)).next_q_delta

        obs, _, base_terminated, truncated, info = self.base_env.step(action)
        curr_q = np.asarray(info["q"], dtype=float)
        curr_dq = np.asarray(info["dq"], dtype=float)
        curr_pose6 = compute_ee_pose6(curr_q)
        action_arr = np.asarray(action, dtype=float)
        q_error_norm = float(np.linalg.norm(goal_q - curr_q))
        prev_q_error_norm = float(np.linalg.norm(goal_q - prev_q))
        action_norm = float(np.linalg.norm(action_arr))
        dq_norm = float(np.linalg.norm(curr_dq))
        nearest_q_dist = float(np.min(np.linalg.norm(self.route.q_goals - curr_q, axis=1)))
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

        waypoint_success = bool(
            ready_now
            and self._ready_streak >= self.config.base_env_config.termination_config.success_dwell_steps
        )
        sequence_success = False
        terminated = False
        reason = "running"
        if waypoint_success:
            self._completed_waypoints += 1
            if target_index >= self._last_route_index:
                sequence_success = True
                terminated = True
                reason = "route_sequence_success"
            else:
                self._advance_target(target_index + 1)
                if self.sequence_config.reset_ready_streak_on_advance:
                    self._ready_streak = 0
                obs = self.base_env.current_observation()

        if bool(base_terminated) and not terminated:
            base_reason = str(info.get("reason", ""))
            if base_reason != "success":
                terminated = True
                reason = base_reason
        if bool(truncated):
            reason = "max_steps"

        info.update(
            self._route_info(
                reset_mode=str(info.get("route_reset_mode", "")),
                waypoint_success=waypoint_success,
                sequence_success=sequence_success,
            )
        )
        info.update(
            {
                "success": bool(sequence_success),
                "route_ready": ready_now,
                "route_ready_streak": int(self._ready_streak),
                "route_q_error_norm": q_error_norm,
                "route_orientation_hit": bool(float(info["orientation_error_norm"]) <= self.config.reward_config.route_ready_ori_threshold_rad),
                "route_regression": bool(q_error_norm > prev_q_error_norm),
                "nearest_route_q_distance": nearest_q_dist,
                "reward_components": components,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reason": reason,
            }
        )
        self._prev_info = dict(info)
        return self._augment_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def _augment_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        target = self.route.waypoint(self._current_route_index)
        return augment_route_observation(
            obs,
            q=self.base_env._q,  # noqa: SLF001
            q_goal=target.q_goal,
            route_tangent_q=self.route.waypoint(max(self._current_route_index - 1, 0)).next_q_delta,
            route_index=self._current_route_index,
            max_route_index=len(self.route) - 1,
            route_progress_m=target.route_progress_m,
            total_route_progress_m=self.route.waypoint(len(self.route) - 1).route_progress_m,
            joint_specs=self.config.base_env_config.joint_specs,
            enabled=self.config.observation_config.include_route_keys,
        )

    def _advance_target(self, next_route_index: int) -> None:
        self._current_route_index = int(next_route_index)
        self.base_env._goal_q = self.route.waypoint(self._current_route_index).q_goal.copy()  # noqa: SLF001
        self.base_env._goal_pose6 = self.route.waypoint(self._current_route_index).ee_target_pose6.copy()  # noqa: SLF001
        self.base_env._capture_entry_metrics()  # noqa: SLF001

    def _route_info(self, *, reset_mode: str, waypoint_success: bool, sequence_success: bool) -> dict[str, Any]:
        target = self.route.waypoint(self._current_route_index)
        return {
            "route_index": int(self._current_route_index),
            "start_route_index": int(self._start_route_index),
            "last_route_index": int(self._last_route_index),
            "route_reset_mode": reset_mode,
            "route_progress_m": target.route_progress_m,
            "route_chunk_id": target.chunk_id,
            "route_sequence_mode": True,
            "route_waypoint_success": bool(waypoint_success),
            "route_sequence_success": bool(sequence_success),
            "route_completed_waypoints": int(self._completed_waypoints),
            "route_sequence_length": int(max(self._last_route_index - self._start_route_index, 1)),
        }

    def close(self) -> None:
        close_fn = getattr(self.base_env, "close", None)
        if callable(close_fn):
            close_fn()
