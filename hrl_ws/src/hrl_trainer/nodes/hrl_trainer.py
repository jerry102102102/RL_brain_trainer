"""ROS 2 node orchestrating hierarchical RL training."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from std_srvs.srv import Trigger
from torch.utils.tensorboard import SummaryWriter

from hrl_brain_trainer import HRLConfig, HierarchicalRLController, OptionSpec
from hrl_trainer.config import OptionConfig, TrainerConfig


class HRLTrainerNode(Node):
    """Trainer node that wraps :class:`HierarchicalRLController` in ROS 2."""

    def __init__(self) -> None:
        super().__init__("hrl_trainer")
        self._declare_default_parameters()
        self.config = TrainerConfig.from_node(self)
        self.training_enabled = self.config.training_mode.lower() == "train"
        share_dir = Path(get_package_share_directory("hrl_trainer"))
        self.log_dir = self._resolve_path(self.config.log_dir, share_dir)
        self.checkpoint_dir = self._resolve_path(self.config.checkpoint_dir, share_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Optional[SummaryWriter] = None
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.goal_lower = np.asarray(self.config.goal_lower, dtype=np.float64)
        self.goal_upper = np.asarray(self.config.goal_upper, dtype=np.float64)
        self._checkpoint_candidates = ["latest.pt", "final.pt"]
        option_specs = [
            OptionSpec(
                name=opt.name,
                goal_offset_scale=opt.frac,
                duration_scale=max(opt.tau, 1e-3),
                dmp_basis_id=idx,
                safety_profile="default",
            )
            for idx, opt in enumerate(self.config.option_set)
        ]
        horizon_max = max(opt.horizon for opt in self.config.option_set)
        controller_cfg = HRLConfig(
            n_joints=self.config.n_joints,
            state_dim=self.config.observation_dim,
            options=option_specs,
            horizon_steps=horizon_max,
            dt=1.0 / max(self.config.control_rate, 1e-6),
            gamma=self.config.hyperparams.gamma,
            lr=self.config.hyperparams.lr,
            batch_size=self.config.hyperparams.batch_size,
            replay_capacity=self.config.hyperparams.replay_capacity,
            eps_start=self.config.hyperparams.eps_start,
            eps_end=self.config.hyperparams.eps_end,
            eps_decay_steps=self.config.hyperparams.eps_decay_steps,
            tau=self.config.hyperparams.tau,
            target_update=self.config.hyperparams.target_update,
            target_update_interval=self.config.hyperparams.target_update_interval,
            double_dqn=self.config.hyperparams.double_dqn,
            seed=self.config.seed,
        )
        self.controller = HierarchicalRLController(controller_cfg)
        self.controller.set_joint_limits(self.goal_lower, self.goal_upper)
        self.controller.reset_norm()
        self.rng = np.random.default_rng(self.config.seed)
        self._maybe_load_checkpoint()

        self.checkpoint_interval = max(1, self.config.runtime.checkpoint_interval)
        self.summary_flush_interval = max(1, self.config.summary_flush_interval)

        qos = QoSProfile(depth=20)
        self.u_cmd_pub = self.create_publisher(Float64MultiArray, "/hrl/u_cmd", qos)
        self.goal_pub = self.create_publisher(Float64MultiArray, "/hrl/goal", qos)
        self.option_pub = self.create_publisher(String, self.config.option_debug_topic, qos)
        self.obs_sub = self.create_subscription(Float64MultiArray, "/hrl/obs", self._obs_cb, qos)
        self.reward_sub = self.create_subscription(Float32, "/hrl/reward", self._reward_cb, qos)
        self.done_sub = self.create_subscription(Bool, "/hrl/done", self._done_cb, qos)
        self.reset_client = self.create_client(Trigger, "/hrl/reset")

        self.current_state: Optional[np.ndarray] = None
        self.current_goal = np.zeros(self.config.n_joints, dtype=np.float64)
        self.pending_reset: bool = False
        self.reset_future = None
        self.awaiting_first_obs = True
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_index = 0
        self.success_count = 0
        self.total_reward_window: Deque[float] = deque(maxlen=self.config.summary_window)
        self.current_plan: List[np.ndarray] = []
        self.pending_done: bool = False
        self.awaiting_transition: bool = False
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.accumulated_reward = 0.0
        self.training_updates = 0

        timer_period = 1.0 / max(self.config.control_rate, 1e-6)
        self.timer = self.create_timer(timer_period, self._on_timer)
        self._ensure_reset()
        self.get_logger().info("HRL trainer node ready (mode=%s).", "train" if self.training_enabled else "eval")

    # ------------------------------------------------------------------
    def _declare_default_parameters(self) -> None:
        defaults = [
            ("training_mode", "train"),
            ("n_joints", 4),
            ("observation_dim", 16),
            ("control_rate", 20.0),
            ("command_interface", "position"),
            ("gamma", 0.99),
            ("batch_size", 64),
            ("replay_capacity", 50000),
            ("eps_start", 1.0),
            ("eps_end", 0.05),
            ("eps_decay_steps", 5000),
            ("lr", 5e-4),
            ("tau", 0.01),
            ("target_update", "soft"),
            ("target_update_interval", 1000),
            ("double_dqn", True),
            ("total_training_steps", 20000),
            ("warmup_steps", 1000),
            ("log_interval", 100),
            ("checkpoint_interval", 2000),
            ("gradient_updates_per_step", 1),
            ("goal_lower", []),
            ("goal_upper", []),
            ("success_tolerance", 0.02),
            ("max_episode_steps", 400),
            ("option_names", []),
            ("option_fracs", []),
            ("option_taus", []),
            ("option_horizons", []),
            ("log_dir", "logs"),
            ("checkpoint_dir", "checkpoints"),
            ("option_debug_topic", "/hrl/option_debug"),
            ("use_tensorboard", True),
            ("summary_flush_interval", 50),
            ("summary_window", 200),
            ("seed", 1),
            ("reward_clip", 20.0),
        ]
        for name, value in defaults:
            self.declare_parameter(name, value)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(path: Path, base: Path) -> Path:
        if path.is_absolute():
            return path
        return (base / path).resolve()

    # ------------------------------------------------------------------
    def _maybe_load_checkpoint(self) -> None:
        for name in self._checkpoint_candidates:
            path = self.checkpoint_dir / name
            if not path.exists():
                continue
            try:
                state = torch.load(path, map_location=self.controller.device)
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().warning("Failed to load checkpoint %s: %s", path, exc)
                return
            try:
                self.controller.policy_net.load_state_dict(state["policy_state"])
                self.controller.target_net.load_state_dict(state["target_state"])
                self.controller.optimizer.load_state_dict(state["optimizer_state"])
                self.controller.total_env_steps = int(state.get("total_env_steps", 0))
                self.controller.training_steps = int(state.get("training_steps", 0))
                norm_state = state.get("state_normalizer")
                if norm_state and self.controller.state_normalizer is not None:
                    self.controller.state_normalizer.load_state_dict(norm_state)
                self.get_logger().info("Loaded checkpoint from %s", path)
                return
            except Exception as exc:  # pragma: no cover - defensive
                self.get_logger().warning("Checkpoint %s could not be applied: %s", path, exc)
                return

    # ------------------------------------------------------------------
    def _ensure_reset(self) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("Waiting for /hrl/reset service...")
        self._request_reset()

    # ------------------------------------------------------------------
    def _request_reset(self) -> None:
        self.pending_reset = True
        self.awaiting_first_obs = True
        self.current_plan.clear()
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0
        self.episode_reward = 0.0
        self.episode_step = 0
        self.current_goal = self._sample_goal()
        goal_msg = Float64MultiArray()
        goal_msg.data = self.current_goal.tolist()
        self.goal_pub.publish(goal_msg)
        self.controller.set_goal(self.current_goal)
        self.reset_future = self.reset_client.call_async(Trigger.Request())

    # ------------------------------------------------------------------
    def _sample_goal(self) -> np.ndarray:
        return self.rng.uniform(self.goal_lower, self.goal_upper)

    # ------------------------------------------------------------------
    def _reward_cb(self, msg: Float32) -> None:
        self.accumulated_reward += float(msg.data)

    # ------------------------------------------------------------------
    def _done_cb(self, msg: Bool) -> None:
        if msg.data:
            self.pending_done = True

    # ------------------------------------------------------------------
    def _obs_cb(self, msg: Float64MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float64)
        if obs.shape[0] < self.config.observation_dim:
            obs = np.pad(obs, (0, self.config.observation_dim - obs.shape[0]))
        elif obs.shape[0] > self.config.observation_dim:
            obs = obs[: self.config.observation_dim]
        self.current_state = obs
        if self.awaiting_first_obs:
            self.awaiting_first_obs = False
            return
        if self.awaiting_transition and self.last_state is not None and self.last_action is not None:
            raw_obs = obs.copy()
            next_state = self._process_state(obs, update_norm=self.training_enabled)
            reward = float(np.clip(self.accumulated_reward, -self.config.reward_clip, self.config.reward_clip))
            done = bool(self.pending_done)
            self.controller.push_transition(self.last_state, self.last_action, reward, next_state, done)
            if self.training_enabled and self.controller.total_env_steps > self.config.runtime.warmup_steps:
                for _ in range(self.config.runtime.gradient_updates_per_step):
                    info = self.controller.train_step()
                    if info.get("loss") is not None and self.writer is not None:
                        self.writer.add_scalar("train/loss", info["loss"], self.controller.training_steps)
                self.training_updates += 1
            self.episode_reward += reward
            self.accumulated_reward = 0.0
            self.pending_done = False
            self.awaiting_transition = False
            self.last_state = None
            self.last_action = None
            if done or self.episode_step >= self.config.max_episode_steps:
                self._finalize_episode(raw_obs)

    # ------------------------------------------------------------------
    def _process_state(self, obs: np.ndarray, update_norm: bool) -> np.ndarray:
        return self.controller._process_state(obs, self.current_goal, update_norm=update_norm)

    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if self.pending_reset:
            if self.reset_future is not None and self.reset_future.done():
                try:
                    result = self.reset_future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    self.get_logger().error("Reset request failed: %s", exc)
                    result = None
                if result is not None:
                    if result.success:
                        self.get_logger().debug("Environment reset acknowledged: %s", result.message)
                    else:
                        self.get_logger().warn("Reset failed: %s", result.message)
                self.pending_reset = False
                self.accumulated_reward = 0.0
            else:
                return
        if self.awaiting_first_obs or self.current_state is None:
            return
        if not self.current_plan:
            self._select_option()
            if not self.current_plan:
                return
        command = self.current_plan.pop(0)
        msg = Float64MultiArray()
        msg.data = command.tolist()
        self.u_cmd_pub.publish(msg)
        self.controller.total_env_steps += 1
        if self.training_enabled:
            self.controller._update_epsilon()
        self.episode_step += 1
        if not self.current_plan:
            self.awaiting_transition = True
        if self.episode_step >= self.config.max_episode_steps:
            self.pending_done = True

        if (
            self.training_enabled
            and self.controller.total_env_steps > 0
            and self.controller.total_env_steps % self.checkpoint_interval == 0
        ):
            self._save_checkpoint("latest.pt")

        if (
            self.writer is not None
            and self.controller.total_env_steps > 0
            and self.controller.total_env_steps % self.summary_flush_interval == 0
        ):
            self.writer.flush()

        if self.training_enabled and self.controller.total_env_steps >= self.config.runtime.total_training_steps:
            self.training_enabled = False
            self.get_logger().info("Reached training step budget. Switching to inference mode.")
            self._save_checkpoint("final.pt")

    # ------------------------------------------------------------------
    def _select_option(self) -> None:
        state = self._process_state(self.current_state, update_norm=self.training_enabled)
        warmup = self.controller.total_env_steps < self.config.runtime.warmup_steps
        if self.training_enabled and warmup:
            action = self.rng.integers(0, len(self.config.option_set))
        else:
            action = self.controller.select_option(state, explore=self.training_enabled)
        self.last_state = state
        self.last_action = int(action)
        macro = self.config.option_set[int(action)]
        self.current_plan = self._generate_plan(int(action), macro)
        if self.option_pub.get_subscription_count() > 0:
            msg = String()
            msg.data = macro.name
            self.option_pub.publish(msg)

    # ------------------------------------------------------------------
    def _generate_plan(self, action: int, macro: OptionConfig) -> List[np.ndarray]:
        positions = self._extract_joint_positions(self.current_state)
        traj, _ = self.controller.option_to_dmp(
            action,
            positions,
            self.current_goal,
            T=max(macro.tau, 1e-3),
            n_steps=max(macro.horizon, 2),
            return_vel=True,
        )
        plan: List[np.ndarray] = []
        for idx in range(1, traj.shape[0]):
            plan.append(np.asarray(traj[idx], dtype=np.float64))
        return plan

    # ------------------------------------------------------------------
    def _extract_joint_positions(self, obs: np.ndarray) -> np.ndarray:
        q_norm = obs[: self.config.n_joints]
        lower = self.goal_lower
        upper = self.goal_upper
        span = np.maximum(upper - lower, 1e-6)
        return lower + (q_norm + 1.0) * 0.5 * span

    # ------------------------------------------------------------------
    def _finalize_episode(self, final_state: np.ndarray) -> None:
        self.episode_index += 1
        success = self._check_success(final_state)
        if success:
            self.success_count += 1
        self.total_reward_window.append(self.episode_reward)
        success_rate = self.success_count / max(self.episode_index, 1)
        self.get_logger().info(
            "Episode %d finished (reward=%.3f, success_rate=%.2f%%)",
            self.episode_index,
            self.episode_reward,
            success_rate * 100.0,
        )
        if self.writer is not None:
            self.writer.add_scalar("episode/reward", self.episode_reward, self.episode_index)
            self.writer.add_scalar("episode/success_rate", success_rate, self.episode_index)
            self.writer.add_scalar("episode/length", self.episode_step, self.episode_index)
        self._request_reset()

    # ------------------------------------------------------------------
    def _check_success(self, state: np.ndarray) -> bool:
        positions = self._extract_joint_positions(state)
        max_error = float(np.max(np.abs(positions - self.current_goal)))
        return max_error < self.config.success_tolerance

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        data = {
            "controller_cfg": asdict(self.controller.cfg),
            "policy_state": self.controller.policy_net.state_dict(),
            "target_state": self.controller.target_net.state_dict(),
            "optimizer_state": self.controller.optimizer.state_dict(),
            "total_env_steps": self.controller.total_env_steps,
            "training_steps": self.controller.training_steps,
            "state_normalizer": None,
        }
        if self.controller.state_normalizer is not None:
            data["state_normalizer"] = self.controller.state_normalizer.state_dict()
        torch.save(data, path)
        self.get_logger().info("Saved checkpoint to %s", path)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = HRLTrainerNode()
    try:
        rclpy.spin(node)
    finally:
        if node.writer is not None:
            node.writer.close()
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["HRLTrainerNode", "main"]
