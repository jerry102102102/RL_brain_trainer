"""Pure forward-kinematics Gymnasium environment for Phase 1/1B."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from .curriculum import PointCurriculumConfig
from .reset_samplers import DockResetConfig, RouteResetConfig, sample_approach_reset, sample_dock_reset
from .reward_approach import ApproachRewardConfig, compute_approach_reward
from .reward_dock import DockRewardConfig, compute_dock_reward
from ..kinematics.fk_interface import compute_ee_pose6, sample_reachable_target
from ..kinematics.joint_limits import (
    JointSpec,
    clip_joint_configuration,
    default_joint_specs,
    delta_limits,
    joint_limit_margin,
    sample_joint_configuration,
)
from ..kinematics.pose_utils import l2_norm, pose_error_components
from .observation_builder import ObservationBuilderConfig, build_observation
from .spaces import EnvBase, build_action_space, build_observation_space
from .termination import TerminationConfig, evaluate_termination
from ..bridge.bridge_reset_samplers import BridgeResetConfig, load_bridge_handoff_states, sample_bridge_reset
from ..bridge.reward_bridge import BridgeRewardConfig, compute_bridge_reward
from ..dock_coarse.reward_dock_coarse import DockCoarseRewardConfig, compute_dock_coarse_reward


@dataclass(frozen=True)
class Phase1EnvConfig:
    mode_name: str = "approach"
    n_joints: int = 7
    joint_specs: tuple[JointSpec, ...] = field(default_factory=default_joint_specs)
    goal_sample_margin_fraction: float = 0.10
    start_sample_margin_fraction: float = 0.20
    action_delta_scale: float = 1.0
    dynamic_action_delta_scale_enabled: bool = False
    dynamic_action_delta_scale_near_pos_threshold_m: float = 0.0
    dynamic_action_delta_scale_far_pos_threshold_m: float = 0.0
    dynamic_action_delta_scale_near_multiplier: float = 1.0
    dynamic_action_delta_scale_far_multiplier: float = 1.0
    dock_action_delta_scale: float = 0.0
    dock_residual_action_limit: float = 1.0
    dock_delta_q_change_limit_scale: float = 0.0
    dock_dynamic_action_limit_near_pos_threshold_m: float = 0.0
    dock_dynamic_action_limit_far_pos_threshold_m: float = 0.0
    dock_dynamic_residual_action_limit_near: float = 1.0
    dock_dynamic_residual_action_limit_far: float = 1.0
    dock_dynamic_delta_q_change_limit_scale_near: float = 0.0
    dock_dynamic_delta_q_change_limit_scale_far: float = 0.0
    episode_length: int = 75
    dwell_steps_target: int = 3
    curriculum_config: PointCurriculumConfig = field(default_factory=PointCurriculumConfig)
    workspace_stage_sampling: dict[str, Any] = field(default_factory=dict)
    reward_config: ApproachRewardConfig = field(default_factory=ApproachRewardConfig)
    dock_reward_config: DockRewardConfig = field(default_factory=DockRewardConfig)
    dock_coarse_reward_config: DockCoarseRewardConfig = field(default_factory=DockCoarseRewardConfig)
    dock_reset_config: DockResetConfig = field(default_factory=DockResetConfig)
    route_reset_config: RouteResetConfig = field(default_factory=RouteResetConfig)
    bridge_reward_config: BridgeRewardConfig = field(default_factory=BridgeRewardConfig)
    bridge_reset_config: BridgeResetConfig = field(default_factory=BridgeResetConfig)
    termination_config: TerminationConfig = field(default_factory=TerminationConfig)
    observation_config: ObservationBuilderConfig = field(default_factory=ObservationBuilderConfig)


class ArmKinematicEnv(EnvBase):
    """Forward-kinematics environment with normalized delta-joint actions."""

    metadata = {"render_modes": []}

    def __init__(self, config: Phase1EnvConfig | None = None) -> None:
        self.config = config or Phase1EnvConfig()
        if len(self.config.joint_specs) != self.config.n_joints:
            raise ValueError("joint_specs length must match n_joints")
        self.action_space = build_action_space(self.config.n_joints)
        self.observation_space = build_observation_space(self.config.n_joints)
        self._rng = np.random.default_rng(0)
        self._episode_step = 0
        self._dwell_count = 0
        self._near_goal_entry_count = 0
        self._near_goal_drift_count = 0
        self._pre_near_goal_hit = False
        self._near_goal_hit = False
        self._min_pos_error = float("inf")
        self._q = np.zeros(self.config.n_joints, dtype=float)
        self._dq = np.zeros(self.config.n_joints, dtype=float)
        self._prev_action = np.zeros(self.config.n_joints, dtype=float)
        self._entry_position_error_norm = 0.0
        self._entry_orientation_error_norm = 0.0
        self._entry_action_l2 = 0.0
        self._entry_dq_norm = 0.0
        self._goal_q = np.zeros(self.config.n_joints, dtype=float)
        self._goal_pose6 = np.zeros(6, dtype=float)
        self._ee_pose6 = compute_ee_pose6(self._q)
        self._curriculum_stage_index = 0
        self._policy_mode_name = self.config.mode_name
        self._bridge_handoff_states = load_bridge_handoff_states(self.config.bridge_reset_config)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}
        self._episode_step = 0
        self._dwell_count = 0
        self._near_goal_entry_count = 0
        self._near_goal_drift_count = 0
        self._pre_near_goal_hit = False
        self._near_goal_hit = False
        self._min_pos_error = float("inf")
        self._policy_mode_name = str(opts.get("policy_mode", self.config.mode_name))

        initial_q = opts.get("initial_q")
        initial_dq = opts.get("initial_dq")
        initial_prev_action = opts.get("initial_prev_action")
        goal_q = opts.get("goal_q")
        goal_pose6 = opts.get("goal_pose6")
        reset_sample = None
        bridge_reset_options: dict[str, Any] | None = None

        if initial_q is not None:
            self._q = clip_joint_configuration(np.asarray(initial_q, dtype=float), self.config.joint_specs)
        else:
            if self._policy_mode_name == "bridge":
                bridge_reset_options = sample_bridge_reset(
                    rng=self._rng,
                    records=self._bridge_handoff_states,
                    joint_specs=self.config.joint_specs,
                    config=self.config.bridge_reset_config,
                )
                self._q = clip_joint_configuration(np.asarray(bridge_reset_options["initial_q"], dtype=float), self.config.joint_specs)
                self._dq = np.asarray(bridge_reset_options["initial_dq"], dtype=float)
                self._prev_action = np.asarray(bridge_reset_options["initial_prev_action"], dtype=float)
                self._goal_q = np.asarray(bridge_reset_options["goal_q"], dtype=float)
                self._goal_pose6 = np.asarray(bridge_reset_options["goal_pose6"], dtype=float)
                self._ee_pose6 = compute_ee_pose6(self._q)
                self._capture_entry_metrics()
                observation = self.current_observation()
                info = self._base_info(success=False, terminated=False, truncated=False, reason="reset")
                info.update(
                    {
                        "bridge_source_bucket_type": bridge_reset_options.get("source_bucket_type", "unknown"),
                        "bridge_source_rollout_id": bridge_reset_options.get("source_rollout_id"),
                        "bridge_source_episode_id": bridge_reset_options.get("source_episode_id"),
                        "bridge_source_step": bridge_reset_options.get("source_step"),
                        "bridge_source_position_error": bridge_reset_options.get("source_position_error"),
                        "bridge_source_orientation_error": bridge_reset_options.get("source_orientation_error"),
                        "bridge_source_dwell_count": bridge_reset_options.get("source_dwell_count"),
                        "bridge_source_action_magnitude": bridge_reset_options.get("source_action_magnitude"),
                        "bridge_source_dq_norm": bridge_reset_options.get("source_dq_norm"),
                    }
                )
                return observation, info
            if self._policy_mode_name in {"dock", "dock_coarse"}:
                reset_sample = sample_dock_reset(
                    rng=self._rng,
                    joint_specs=self.config.joint_specs,
                    dock_reset_config=self.config.dock_reset_config,
                    curriculum_config=self.config.curriculum_config,
                    stage_index=self._curriculum_stage_index,
                )
            else:
                reset_sample = sample_approach_reset(
                    rng=self._rng,
                    joint_specs=self.config.joint_specs,
                    curriculum_config=self.config.curriculum_config,
                    stage_index=self._curriculum_stage_index,
                    start_margin_fraction=self.config.start_sample_margin_fraction,
                    goal_margin_fraction=self.config.goal_sample_margin_fraction,
                    route_reset_config=self.config.route_reset_config,
                    workspace_stage_sampling=self.config.workspace_stage_sampling,
                )
            self._q = reset_sample.initial_q
        if initial_dq is not None:
            self._dq = np.asarray(initial_dq, dtype=float)
        elif reset_sample is not None and reset_sample.initial_dq is not None:
            self._dq = np.asarray(reset_sample.initial_dq, dtype=float)
        else:
            self._dq = np.zeros(self.config.n_joints, dtype=float)
        if initial_prev_action is not None:
            self._prev_action = np.asarray(initial_prev_action, dtype=float)
        elif reset_sample is not None and reset_sample.initial_prev_action is not None:
            self._prev_action = np.asarray(reset_sample.initial_prev_action, dtype=float)
        else:
            self._prev_action = np.zeros(self.config.n_joints, dtype=float)
        self._ee_pose6 = compute_ee_pose6(self._q)

        if goal_pose6 is not None:
            self._goal_pose6 = np.asarray(goal_pose6, dtype=float)
            self._goal_q = np.asarray(goal_q, dtype=float) if goal_q is not None else np.zeros(self.config.n_joints, dtype=float)
        elif goal_q is not None:
            self._goal_q = clip_joint_configuration(np.asarray(goal_q, dtype=float), self.config.joint_specs)
            self._goal_pose6 = compute_ee_pose6(self._goal_q)
        elif initial_q is None:
            self._goal_q = reset_sample.goal_q
            self._goal_pose6 = reset_sample.goal_pose6
        else:
            sampled = sample_reachable_target(
                self._rng,
                self.config.joint_specs,
                margin_fraction=self.config.goal_sample_margin_fraction,
            )
            self._goal_q = sampled.q
            self._goal_pose6 = sampled.pose6

        self._capture_entry_metrics()
        observation = self.current_observation()
        info = self._base_info(success=False, terminated=False, truncated=False, reason="reset")
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_arr = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if action_arr.shape != (self.config.n_joints,):
            raise ValueError(f"Expected action shape {(self.config.n_joints,)}, got {action_arr.shape}")
        prev_pose6 = self._ee_pose6.copy()
        prev_action = self._prev_action.copy()
        prev_pos_err, prev_ori_err = pose_error_components(prev_pose6, self._goal_pose6)
        prev_pos_norm = l2_norm(prev_pos_err)
        prev_ori_norm = l2_norm(prev_ori_err)
        dynamic_dock_limit = float(np.clip(self.config.dock_residual_action_limit, 0.0, 1.0))
        dynamic_dq_change_limit_scale = float(max(self.config.dock_delta_q_change_limit_scale, 0.0))
        if self._policy_mode_name in {"dock", "dock_coarse"}:
            dynamic_dock_limit = self._dynamic_dock_residual_action_limit(prev_pos_norm)
            dynamic_dq_change_limit_scale = self._dynamic_dock_delta_q_change_limit_scale(prev_pos_norm)
            dock_limit = dynamic_dock_limit
            action_arr = np.clip(action_arr, -dock_limit, dock_limit)

        curr_in_pre_near_goal = False
        prev_in_near_goal = self._is_near_goal(prev_pos_norm, prev_ori_norm)
        action_delta_scale = float(self.config.action_delta_scale)
        if self._policy_mode_name in {"dock", "dock_coarse"} and self.config.dock_action_delta_scale > 0.0:
            action_delta_scale = float(self.config.dock_action_delta_scale)
        elif self._policy_mode_name not in {"dock", "dock_coarse"}:
            action_delta_scale = self._dynamic_action_delta_scale(prev_pos_norm)
        max_delta_q = delta_limits(self.config.joint_specs) * action_delta_scale
        delta_q_cmd = action_arr * max_delta_q
        if self._policy_mode_name in {"dock", "dock_coarse"} and dynamic_dq_change_limit_scale > 0.0:
            dq_change_limit = max_delta_q * dynamic_dq_change_limit_scale
            delta_q_cmd = self._dq + np.clip(delta_q_cmd - self._dq, -dq_change_limit, dq_change_limit)
            delta_q_cmd = np.clip(delta_q_cmd, -max_delta_q, max_delta_q)
        q_next = clip_joint_configuration(self._q + delta_q_cmd, self.config.joint_specs)
        dq_next = q_next - self._q
        delta_q_change_l2 = float(np.linalg.norm(dq_next - self._dq))
        ee_pose_next = compute_ee_pose6(q_next)

        curr_pos_err, curr_ori_err = pose_error_components(ee_pose_next, self._goal_pose6)
        curr_pos_norm = l2_norm(curr_pos_err)
        curr_ori_norm = l2_norm(curr_ori_err)
        curr_in_pre_near_goal = self._is_pre_near_goal(curr_pos_norm, curr_ori_norm)
        curr_in_near_goal = self._is_near_goal(curr_pos_norm, curr_ori_norm)
        self._min_pos_error = min(self._min_pos_error, curr_pos_norm)

        if curr_in_pre_near_goal:
            self._pre_near_goal_hit = True
        if curr_in_near_goal and not prev_in_near_goal:
            self._near_goal_entry_count += 1
        if curr_in_near_goal:
            self._dwell_count += 1
        else:
            self._dwell_count = 0
        if prev_in_near_goal and curr_pos_norm > prev_pos_norm:
            self._near_goal_drift_count += 1

        termination = evaluate_termination(
            step_count=self._episode_step + 1,
            pos_error_norm=curr_pos_norm,
            ori_error_norm=curr_ori_norm,
            dwell_count=self._dwell_count,
            config=self.config.termination_config,
        )
        reward_kwargs = {
            "prev_pose6": prev_pose6,
            "curr_pose6": ee_pose_next,
            "goal_pose6": self._goal_pose6,
            "action": action_arr,
            "prev_action": prev_action,
            "prev_in_near_goal": prev_in_near_goal,
            "curr_in_near_goal": curr_in_near_goal,
            "dwell_count": self._dwell_count,
            "near_goal_entry_count": self._near_goal_entry_count,
            "near_goal_drift_count": self._near_goal_drift_count,
            "joint_limit_margin_min": float(np.min(joint_limit_margin(q_next, self.config.joint_specs))),
            "success": bool(termination["success"]),
        }
        if self._policy_mode_name == "dock":
            reward, reward_components = compute_dock_reward(
                config=self.config.dock_reward_config,
                delta_q_change_l2=delta_q_change_l2,
                dq_norm=float(np.linalg.norm(dq_next)),
                entry_pos_error_norm=self._entry_position_error_norm,
                entry_ori_error_norm=self._entry_orientation_error_norm,
                entry_action_l2=self._entry_action_l2,
                entry_dq_norm=self._entry_dq_norm,
                **reward_kwargs,
            )
        elif self._policy_mode_name == "dock_coarse":
            reward, reward_components = compute_dock_coarse_reward(
                prev_pose6=prev_pose6,
                curr_pose6=ee_pose_next,
                goal_pose6=self._goal_pose6,
                action=action_arr,
                prev_action=prev_action,
                prev_in_near_goal=prev_in_near_goal,
                curr_in_near_goal=curr_in_near_goal,
                dwell_count=self._dwell_count,
                joint_limit_margin_min=float(np.min(joint_limit_margin(q_next, self.config.joint_specs))),
                success=bool(termination["success"]),
                dq_norm=float(np.linalg.norm(dq_next)),
                prev_dq_norm=float(np.linalg.norm(self._dq)),
                config=self.config.dock_coarse_reward_config,
            )
        elif self._policy_mode_name == "bridge":
            reward, reward_components = compute_bridge_reward(
                prev_pose6=prev_pose6,
                curr_pose6=ee_pose_next,
                goal_pose6=self._goal_pose6,
                action=action_arr,
                prev_action=prev_action,
                dq_norm=float(np.linalg.norm(dq_next)),
                joint_limit_margin_min=float(np.min(joint_limit_margin(q_next, self.config.joint_specs))),
                config=self.config.bridge_reward_config,
            )
            if (
                self.config.bridge_reward_config.terminate_on_leave_near_goal
                and curr_pos_norm > self.config.bridge_reward_config.position_keep_radius_m
            ):
                termination = {
                    "success": False,
                    "terminated": True,
                    "truncated": False,
                    "reason": "bridge_left_near_goal",
                }
        else:
            reward, reward_components = compute_approach_reward(
                curr_in_pre_near_goal=curr_in_pre_near_goal,
                dq_norm=float(np.linalg.norm(dq_next)),
                prev_dq_norm=float(np.linalg.norm(self._dq)),
                config=self.config.reward_config,
                **reward_kwargs,
            )

        self._episode_step += 1
        self._q = q_next
        self._dq = dq_next
        self._prev_action = action_arr
        self._ee_pose6 = ee_pose_next
        if curr_in_near_goal:
            self._near_goal_hit = True

        observation = self.current_observation()
        info = self._base_info(
            success=bool(termination["success"]),
            terminated=bool(termination["terminated"]),
            truncated=bool(termination["truncated"]),
            reason=str(termination["reason"]),
        )
        info["reward_components"] = reward_components
        info["action_l2"] = float(np.linalg.norm(action_arr))
        info["executed_delta_q_l2"] = float(np.linalg.norm(dq_next))
        info["delta_q_change_l2"] = delta_q_change_l2
        info["dock_action_limit"] = dynamic_dock_limit
        info["dock_delta_q_change_limit_scale"] = dynamic_dq_change_limit_scale
        return observation, float(reward), bool(termination["terminated"]), bool(termination["truncated"]), info

    def _build_observation(self, *, mode_index: int) -> dict[str, np.ndarray]:
        return build_observation(
            q=self._q,
            dq=self._dq,
            prev_action=self._prev_action,
            current_pose6=self._ee_pose6,
            goal_pose6=self._goal_pose6,
            joint_specs=self.config.joint_specs,
            episode_progress=self._episode_step / max(self.config.episode_length, 1),
            dwell_progress=self._dwell_count / max(self.config.dwell_steps_target, 1),
            mode_index=mode_index,
            config=self.config.observation_config,
        )

    def current_observation(self) -> dict[str, np.ndarray]:
        return self._build_observation(mode_index=self._mode_index())

    def _base_info(self, *, success: bool, terminated: bool, truncated: bool, reason: str) -> dict[str, Any]:
        pos_err_vec, ori_err_vec = pose_error_components(self._ee_pose6, self._goal_pose6)
        pos_err_norm = l2_norm(pos_err_vec)
        ori_err_norm = l2_norm(ori_err_vec)
        return {
            "goal_pose6": self._goal_pose6.copy(),
            "goal_q": self._goal_q.copy(),
            "q": self._q.copy(),
            "dq": self._dq.copy(),
            "ee_pose6": self._ee_pose6.copy(),
            "position_error_norm": pos_err_norm,
            "orientation_error_norm": ori_err_norm,
            "position_error_vec": pos_err_vec.copy(),
            "orientation_error_vec": ori_err_vec.copy(),
            "curr_in_pre_near_goal": bool(self._is_pre_near_goal(pos_err_norm, ori_err_norm)),
            "curr_in_near_goal": bool(self._is_near_goal(pos_err_norm, ori_err_norm)),
            "pre_near_goal_hit": bool(self._pre_near_goal_hit),
            "near_goal_hit": bool(self._near_goal_hit),
            "dwell_count": int(self._dwell_count),
            "near_goal_entry_count": int(self._near_goal_entry_count),
            "near_goal_drift_count": int(self._near_goal_drift_count),
            "min_position_error": float(self._min_pos_error),
            "entry_position_error_norm": float(self._entry_position_error_norm),
            "entry_orientation_error_norm": float(self._entry_orientation_error_norm),
            "entry_action_l2": float(self._entry_action_l2),
            "entry_dq_norm": float(self._entry_dq_norm),
            "joint_limit_margin_min": float(np.min(joint_limit_margin(self._q, self.config.joint_specs))),
            "success": bool(success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reason": reason,
            "mode_name": self._policy_mode_name,
            "curriculum_stage_index": int(self._curriculum_stage_index),
            "step_count": int(self._episode_step),
            "curriculum_stage_name": (
                self.config.curriculum_config.stages[self._curriculum_stage_index].name
                if self.config.curriculum_config.enabled
                else "random_goal"
            ),
        }

    def _capture_entry_metrics(self) -> None:
        pos_err_vec, ori_err_vec = pose_error_components(self._ee_pose6, self._goal_pose6)
        self._entry_position_error_norm = l2_norm(pos_err_vec)
        self._entry_orientation_error_norm = l2_norm(ori_err_vec)
        self._entry_action_l2 = float(np.linalg.norm(self._prev_action))
        self._entry_dq_norm = float(np.linalg.norm(self._dq))

    def _is_near_goal(self, pos_error_norm: float, ori_error_norm: float) -> bool:
        if pos_error_norm > self.config.reward_config.near_goal_pos_threshold_m:
            return False
        if self.config.reward_config.use_orientation_gate and ori_error_norm > self.config.reward_config.near_goal_ori_threshold_rad:
            return False
        return True

    def _is_pre_near_goal(self, pos_error_norm: float, ori_error_norm: float) -> bool:
        if pos_error_norm > self.config.reward_config.pre_near_goal_pos_threshold_m:
            return False
        if self.config.reward_config.use_orientation_gate and ori_error_norm > self.config.reward_config.near_goal_ori_threshold_rad:
            return False
        return True

    def set_curriculum_stage(self, stage_index: int) -> None:
        if not self.config.curriculum_config.enabled:
            return
        self._curriculum_stage_index = int(np.clip(stage_index, 0, len(self.config.curriculum_config.stages) - 1))

    def get_curriculum_stage(self) -> int:
        return int(self._curriculum_stage_index)

    def set_policy_mode(self, mode_name: str) -> None:
        if mode_name not in {"approach", "bridge", "dock", "dock_coarse"}:
            raise ValueError(f"Unsupported policy mode '{mode_name}'")
        self._policy_mode_name = mode_name

    def apply_dock_training_stage(self, stage_updates: dict[str, Any]) -> None:
        if self.config.mode_name != "dock":
            return

        next_config = self.config
        dock_reset_updates = dict(stage_updates.get("dock_reset", {}))
        if dock_reset_updates:
            next_dock_reset = replace(next_config.dock_reset_config, **dock_reset_updates)
            next_config = replace(next_config, dock_reset_config=next_dock_reset)

        env_updates: dict[str, Any] = {}
        for key in (
            "action_delta_scale",
            "dock_action_delta_scale",
            "dock_residual_action_limit",
            "dock_delta_q_change_limit_scale",
            "dock_dynamic_action_limit_near_pos_threshold_m",
            "dock_dynamic_action_limit_far_pos_threshold_m",
            "dock_dynamic_residual_action_limit_near",
            "dock_dynamic_residual_action_limit_far",
            "dock_dynamic_delta_q_change_limit_scale_near",
            "dock_dynamic_delta_q_change_limit_scale_far",
        ):
            if key in stage_updates:
                env_updates[key] = float(stage_updates[key])
        if env_updates:
            next_config = replace(next_config, **env_updates)

        self.config = next_config

    def _interpolate_dock_control_value(
        self,
        *,
        pos_error_norm: float,
        near_threshold: float,
        far_threshold: float,
        near_value: float,
        far_value: float,
        fallback: float,
    ) -> float:
        if near_threshold <= 0.0 or far_threshold <= near_threshold:
            return float(fallback)
        if pos_error_norm <= near_threshold:
            return float(near_value)
        if pos_error_norm >= far_threshold:
            return float(far_value)
        alpha = (pos_error_norm - near_threshold) / max(far_threshold - near_threshold, 1e-9)
        return float(near_value + alpha * (far_value - near_value))

    def _dynamic_dock_residual_action_limit(self, pos_error_norm: float) -> float:
        limit = self._interpolate_dock_control_value(
            pos_error_norm=pos_error_norm,
            near_threshold=float(self.config.dock_dynamic_action_limit_near_pos_threshold_m),
            far_threshold=float(self.config.dock_dynamic_action_limit_far_pos_threshold_m),
            near_value=float(self.config.dock_dynamic_residual_action_limit_near),
            far_value=float(self.config.dock_dynamic_residual_action_limit_far),
            fallback=float(self.config.dock_residual_action_limit),
        )
        return float(np.clip(limit, 0.0, 1.0))

    def _dynamic_dock_delta_q_change_limit_scale(self, pos_error_norm: float) -> float:
        limit = self._interpolate_dock_control_value(
            pos_error_norm=pos_error_norm,
            near_threshold=float(self.config.dock_dynamic_action_limit_near_pos_threshold_m),
            far_threshold=float(self.config.dock_dynamic_action_limit_far_pos_threshold_m),
            near_value=float(self.config.dock_dynamic_delta_q_change_limit_scale_near),
            far_value=float(self.config.dock_dynamic_delta_q_change_limit_scale_far),
            fallback=float(self.config.dock_delta_q_change_limit_scale),
        )
        return float(max(limit, 0.0))

    def _dynamic_action_delta_scale(self, pos_error_norm: float) -> float:
        base_scale = float(self.config.action_delta_scale)
        if not self.config.dynamic_action_delta_scale_enabled:
            return base_scale
        multiplier = self._interpolate_dock_control_value(
            pos_error_norm=pos_error_norm,
            near_threshold=float(self.config.dynamic_action_delta_scale_near_pos_threshold_m),
            far_threshold=float(self.config.dynamic_action_delta_scale_far_pos_threshold_m),
            near_value=float(self.config.dynamic_action_delta_scale_near_multiplier),
            far_value=float(self.config.dynamic_action_delta_scale_far_multiplier),
            fallback=1.0,
        )
        return float(base_scale * max(multiplier, 0.0))

    def _mode_index(self) -> int:
        if self._policy_mode_name == "approach":
            return 0
        if self._policy_mode_name == "dock":
            return 1
        if self._policy_mode_name == "dock_coarse":
            return 3
        return 2

    def render(self) -> None:  # pragma: no cover - no visual renderer in Phase 1
        return None

    def close(self) -> None:
        return None
