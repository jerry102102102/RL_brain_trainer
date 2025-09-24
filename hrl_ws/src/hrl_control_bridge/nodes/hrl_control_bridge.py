"""ROS 2 bridge node connecting the HRL trainer with ros2_control."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray
from std_srvs.srv import Empty, Trigger

from hrl_control_bridge.bridge_node import BridgeConfig


class HRLControlBridgeNode(Node):
    """Bridge node that exposes the Gazebo manipulator as an RL environment."""

    SUCCESS_REWARD = 10.0

    def __init__(self) -> None:
        super().__init__("hrl_control_bridge")
        self._declare_default_parameters()
        self.config = BridgeConfig.from_node(self)
        self.get_logger().info(f"Loaded bridge configuration for joints: {self.config.joint_names}")
        self._rng = random.Random(self.config.random_seed)
        qos = QoSProfile(depth=10)

        # Publishers
        self.command_pub = self.create_publisher(Float64MultiArray, self.config.command_topic, qos)
        self.obs_pub = self.create_publisher(Float64MultiArray, self.config.obs_topic, qos)
        self.reward_pub = self.create_publisher(Float32, self.config.reward_topic, qos)
        self.done_pub = self.create_publisher(Bool, self.config.done_topic, qos)
        self.slack_pub = self.create_publisher(Float32, self.config.slack_topic, qos)

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, qos)
        self.create_subscription(Float64MultiArray, self.config.goal_topic, self._goal_cb, qos)
        self.create_subscription(Float64MultiArray, self.config.u_cmd_topic, self._command_request_cb, qos)

        # Reset service
        self.create_service(Trigger, self.config.reset_service, self._handle_reset)
        self._reset_clients = [
            self.create_client(Empty, name)
            for name in ("/reset_world", "/gazebo/reset_world", "/reset_simulation", "/gazebo/reset_simulation")
        ]

        # Internal state
        n = len(self.config.joint_names)
        self._joint_indices: Dict[str, int] = {}
        self._joint_positions = np.zeros(n, dtype=np.float64)
        self._joint_velocities = np.zeros(n, dtype=np.float64)
        self._goal = np.zeros(n, dtype=np.float64)
        self._last_command = np.zeros(n, dtype=np.float64)
        self._has_joint_state = False
        self._step_count = 0
        self._prev_distance: Optional[float] = None
        self._slack_accum = 0.0
        self._last_slack = 0.0
        self._success_window: Deque[bool] = deque(maxlen=max(1, self.config.success_window))
        self._qp_infeasible = False
        self._collision_detected = False

        self._timer = self.create_timer(1.0 / max(self.config.control_frequency, 1e-3), self._on_timer)

    # ------------------------------------------------------------------
    def _declare_default_parameters(self) -> None:
        defaults = [
            ("control_frequency", 50.0),
            ("joint_names", []),
            ("command_topic", "/forward_position_controller/commands"),
            ("command_interface", "position"),
            ("u_cmd_topic", "/hrl/u_cmd"),
            ("goal_topic", "/hrl/goal"),
            ("obs_topic", "/hrl/obs"),
            ("reward_topic", "/hrl/reward"),
            ("done_topic", "/hrl/done"),
            ("slack_topic", "/hrl/slack"),
            ("reset_service", "/hrl/reset"),
            ("success_window", 5),
            ("success_position_tolerance", 0.0175),
            ("success_velocity_tolerance", 0.02),
            ("success_slack_accum", 1e-3),
            ("publish_slack", True),
            ("use_cbf", False),
            ("max_episode_steps", 500),
            ("progress_reward_scale", 1.0),
            ("time_penalty", 0.02),
            ("slack_penalty", 5.0),
            ("infeasible_penalty", 5.0),
            ("collision_penalty", 5.0),
            ("joint_position_limits.lower", []),
            ("joint_position_limits.upper", []),
            ("joint_velocity_limits", []),
            ("initial_position_range.lower", []),
            ("initial_position_range.upper", []),
            ("random_seed", 0),
        ]
        for name, value in defaults:
            self.declare_parameter(name, value)

    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name:
            return
        if not self._joint_indices:
            for idx, name in enumerate(msg.name):
                self._joint_indices[name] = idx
            missing = [j for j in self.config.joint_names if j not in self._joint_indices]
            if missing:
                self.get_logger().warning(f"JointState is missing joints: {missing}")
        positions = np.zeros_like(self._joint_positions)
        velocities = np.zeros_like(self._joint_velocities)
        for idx, joint in enumerate(self.config.joint_names):
            source_idx = self._joint_indices.get(joint)
            if source_idx is None:
                continue
            if source_idx < len(msg.position):
                positions[idx] = msg.position[source_idx]
            if source_idx < len(msg.velocity):
                velocities[idx] = msg.velocity[source_idx]
        self._joint_positions = positions
        self._joint_velocities = velocities
        self._has_joint_state = True

    # ------------------------------------------------------------------
    def _goal_cb(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != self._goal.shape[0]:
            self.get_logger().warning(
                "Received goal of incorrect dimension: %d (expected %d)",
                data.shape[0],
                self._goal.shape[0],
            )
            return
        self._goal = data

    # ------------------------------------------------------------------
    def _command_request_cb(self, msg: Float64MultiArray) -> None:
        command = np.asarray(msg.data, dtype=np.float64)
        if command.shape[0] != self._joint_positions.shape[0]:
            self.get_logger().warning(
                "Command length %d does not match number of joints %d",
                command.shape[0],
                self._joint_positions.shape[0],
            )
            return
        clipped = self._clip_command(command)
        self._last_command = clipped
        out = Float64MultiArray()
        out.data = clipped.tolist()
        self.command_pub.publish(out)

    # ------------------------------------------------------------------
    def _clip_command(self, command: np.ndarray) -> np.ndarray:
        if self.config.command_interface.lower().startswith("vel"):
            limits = np.asarray(self.config.joint_velocity_limits, dtype=np.float64)
            return np.clip(command, -limits, limits)
        lower = np.asarray(self.config.joint_position_lower, dtype=np.float64)
        upper = np.asarray(self.config.joint_position_upper, dtype=np.float64)
        return np.clip(command, lower, upper)

    # ------------------------------------------------------------------
    def _handle_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:  # type: ignore[override]
        self.get_logger().info("Received HRL environment reset request.")
        for client in self._reset_clients:
            if client.service_is_ready():
                client.call_async(Empty.Request())
        initial_position = self._sample_initial_position()
        if self.config.command_interface.lower().startswith("vel"):
            command = np.zeros_like(initial_position)
        else:
            command = initial_position
        msg = Float64MultiArray()
        msg.data = command.tolist()
        self.command_pub.publish(msg)
        self._joint_positions = initial_position
        self._joint_velocities.fill(0.0)
        self._step_count = 0
        self._prev_distance = None
        self._slack_accum = 0.0
        self._last_slack = 0.0
        self._success_window.clear()
        self._qp_infeasible = False
        self._collision_detected = False
        response.success = True
        response.message = "HRL bridge reset complete."
        return response

    # ------------------------------------------------------------------
    def _sample_initial_position(self) -> np.ndarray:
        lower = np.asarray(self.config.initial_position_lower, dtype=np.float64)
        upper = np.asarray(self.config.initial_position_upper, dtype=np.float64)
        samples = np.array([self._rng.uniform(lo, hi) for lo, hi in zip(lower, upper)], dtype=np.float64)
        return samples

    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if not self._has_joint_state:
            return
        q = self._joint_positions.copy()
        dq = self._joint_velocities.copy()
        goal = self._goal.copy()
        q_norm = self._normalize_positions(q)
        dq_clip = self._clip_velocity(dq)
        state_vec = np.concatenate([q_norm, dq_clip, goal - q, goal], dtype=np.float64)
        obs_msg = Float64MultiArray()
        obs_msg.data = state_vec.tolist()
        self.obs_pub.publish(obs_msg)

        distance = float(np.linalg.norm(goal - q))
        progress = 0.0
        if self._prev_distance is not None:
            progress = (self._prev_distance - distance) * self.config.progress_reward_scale
        self._prev_distance = distance
        reward = progress - self.config.time_penalty
        reward -= self.config.slack_penalty * abs(self._last_slack)
        self._slack_accum += abs(self._last_slack)
        if self._qp_infeasible:
            reward -= self.config.infeasible_penalty
        if self._collision_detected:
            reward -= self.config.collision_penalty

        success = self._update_success_condition(q, dq)
        done = False
        if success:
            reward += self.SUCCESS_REWARD
            done = True
        elif self._step_count >= self.config.max_episode_steps:
            done = True
        elif self._qp_infeasible or self._collision_detected:
            done = True

        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)

        done_msg = Bool()
        done_msg.data = bool(done)
        self.done_pub.publish(done_msg)

        if self.config.publish_slack:
            slack_msg = Float32()
            slack_msg.data = float(self._last_slack)
            self.slack_pub.publish(slack_msg)

        self._step_count += 1
        if done:
            # Freeze until external reset.
            self._step_count = self.config.max_episode_steps

    # ------------------------------------------------------------------
    def _normalize_positions(self, q: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.config.joint_position_lower, dtype=np.float64)
        upper = np.asarray(self.config.joint_position_upper, dtype=np.float64)
        span = np.maximum(upper - lower, 1e-6)
        normalized = 2.0 * (q - lower) / span - 1.0
        return np.clip(normalized, -1.0, 1.0)

    # ------------------------------------------------------------------
    def _clip_velocity(self, dq: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.config.joint_velocity_limits, dtype=np.float64)
        limits = np.maximum(limits, 1e-6)
        return np.clip(dq, -limits, limits)

    # ------------------------------------------------------------------
    def _update_success_condition(self, q: np.ndarray, dq: np.ndarray) -> bool:
        pos_err = np.max(np.abs(q - self._goal))
        vel_norm = np.max(np.abs(dq))
        within_pos = bool(pos_err < self.config.success_position_tolerance)
        within_vel = bool(vel_norm < self.config.success_velocity_tolerance)
        self._success_window.append(within_pos and within_vel)
        stable = len(self._success_window) == self._success_window.maxlen and all(self._success_window)
        if stable and self._slack_accum <= self.config.success_slack_accum:
            return True
        return False


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = HRLControlBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["HRLControlBridgeNode", "main"]
