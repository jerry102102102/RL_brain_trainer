"""ROS2 runtime adapter for V5.1 real Gazebo control/readback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class JointStateFrame:
    names: list[str]
    position: list[float]
    velocity: list[float]
    stamp_ns: int


class JointRuntimeIO(Protocol):
    def publish_joint_target(self, joint_names: list[str], positions: np.ndarray, duration_s: float) -> None: ...

    def wait_for_joint_state(self, timeout_s: float) -> JointStateFrame: ...


class ROS2JointRuntimeIO:
    """ROS2 I/O backend using JointTrajectory publish + /joint_states subscribe."""

    def __init__(self, trajectory_topic: str, joint_state_topic: str) -> None:
        try:
            import rclpy
            from trajectory_msgs.msg import JointTrajectory
            from sensor_msgs.msg import JointState
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in ROS2 env
            raise RuntimeError("ROS2 dependencies are missing; source ROS2 and install msgs") from exc

        self._rclpy = rclpy
        self._JointTrajectory = JointTrajectory
        self._JointState = JointState

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self._node = rclpy.create_node("v5_1_runtime_adapter")
        self._pub = self._node.create_publisher(JointTrajectory, trajectory_topic, 10)
        self._latest: JointStateFrame | None = None

        def _on_joint_state(msg: Any) -> None:
            stamp_ns = int(time.time_ns())
            if getattr(msg, "header", None) and msg.header.stamp is not None:
                stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            self._latest = JointStateFrame(
                names=list(msg.name),
                position=[float(x) for x in msg.position],
                velocity=[float(x) for x in msg.velocity] if msg.velocity else [0.0] * len(msg.position),
                stamp_ns=stamp_ns,
            )

        self._sub = self._node.create_subscription(JointState, joint_state_topic, _on_joint_state, 10)

    def publish_joint_target(self, joint_names: list[str], positions: np.ndarray, duration_s: float) -> None:
        from trajectory_msgs.msg import JointTrajectoryPoint

        msg = self._JointTrajectory()
        msg.joint_names = list(joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions.tolist()]
        sec = int(max(0.0, float(duration_s)))
        nsec = int((max(0.0, float(duration_s)) - sec) * 1_000_000_000)
        pt.time_from_start.sec = sec
        pt.time_from_start.nanosec = nsec
        msg.points = [pt]
        self._pub.publish(msg)

    def wait_for_joint_state(self, timeout_s: float) -> JointStateFrame:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            if self._latest is not None:
                return self._latest
        raise TimeoutError(f"timeout waiting /joint_states after {timeout_s:.2f}s")

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()


class RuntimeROS2Adapter:
    """High-level runtime adapter exposing step(cmd)->obs_next for e2e pipeline."""

    def __init__(
        self,
        io: JointRuntimeIO,
        joint_names: list[str],
        command_duration_s: float = 0.2,
        settle_timeout_s: float = 0.8,
        settle_hold_s: float = 0.12,
        settle_position_epsilon: float = 1e-4,
        min_command_l2: float = 1e-4,
        no_effect_l2: float = 1e-4,
        no_effect_ratio: float = 0.1,
    ) -> None:
        self.io = io
        self.joint_names = list(joint_names)
        self.command_duration_s = float(command_duration_s)
        self.settle_timeout_s = float(settle_timeout_s)
        self.settle_hold_s = max(0.0, float(settle_hold_s))
        self.settle_position_epsilon = max(0.0, float(settle_position_epsilon))
        self.min_command_l2 = max(0.0, float(min_command_l2))
        self.no_effect_l2 = max(0.0, float(no_effect_l2))
        self.no_effect_ratio = max(0.0, float(no_effect_ratio))

    @classmethod
    def from_ros2(
        cls,
        joint_names: list[str],
        trajectory_topic: str = "/arm_controller/joint_trajectory",
        joint_state_topic: str = "/joint_states",
        command_duration_s: float = 0.2,
        settle_timeout_s: float = 0.8,
    ) -> "RuntimeROS2Adapter":
        return cls(
            io=ROS2JointRuntimeIO(trajectory_topic=trajectory_topic, joint_state_topic=joint_state_topic),
            joint_names=joint_names,
            command_duration_s=command_duration_s,
            settle_timeout_s=settle_timeout_s,
        )

    def _extract_q(self, frame: JointStateFrame) -> np.ndarray:
        idx = {name: i for i, name in enumerate(frame.names)}
        missing = [name for name in self.joint_names if name not in idx]
        if missing:
            raise ValueError(f"/joint_states missing joints: {missing}")
        return np.asarray([frame.position[idx[name]] for name in self.joint_names], dtype=float)

    def _read_frame(self, timeout_s: float | None = None) -> JointStateFrame:
        frame = self.io.wait_for_joint_state(timeout_s=self.settle_timeout_s if timeout_s is None else timeout_s)
        _ = self._extract_q(frame)
        return frame

    def read_q(self, timeout_s: float | None = None) -> np.ndarray:
        frame = self._read_frame(timeout_s=timeout_s)
        return self._extract_q(frame)

    def _wait_for_fresh_frame(self, older_than_stamp_ns: int, deadline: float) -> JointStateFrame:
        latest = None
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            candidate = self._read_frame(timeout_s=remaining)
            latest = candidate
            if candidate.stamp_ns > older_than_stamp_ns:
                return candidate
        if latest is not None:
            return latest
        raise TimeoutError("timed out waiting for fresh joint state frame")

    def _wait_until_settled(self, latest_frame: JointStateFrame, deadline: float) -> JointStateFrame:
        if self.settle_hold_s <= 0.0:
            return latest_frame

        hold_start = None
        q_prev = self._extract_q(latest_frame)
        settled = latest_frame
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                candidate = self._read_frame(timeout_s=remaining)
            except TimeoutError:
                break
            q_now = self._extract_q(candidate)
            delta = float(np.linalg.norm(q_now - q_prev))

            if delta <= self.settle_position_epsilon:
                if hold_start is None:
                    hold_start = time.monotonic()
                if (time.monotonic() - hold_start) >= self.settle_hold_s:
                    settled = candidate
                    break
            else:
                hold_start = None

            q_prev = q_now
            settled = candidate

        return settled

    def step(self, cmd_q: np.ndarray) -> dict[str, Any]:
        cmd_q = np.asarray(cmd_q, dtype=float)
        if cmd_q.shape != (len(self.joint_names),):
            raise ValueError(
                f"cmd_q shape mismatch: expected {(len(self.joint_names),)}, got {tuple(cmd_q.shape)}"
            )

        frame_before = self._read_frame()
        q_before = self._extract_q(frame_before)
        cmd_delta_l2 = float(np.linalg.norm(cmd_q - q_before))

        skipped_publish = cmd_delta_l2 < self.min_command_l2
        if not skipped_publish:
            self.io.publish_joint_target(self.joint_names, cmd_q, self.command_duration_s)

        deadline = time.monotonic() + self.settle_timeout_s
        if skipped_publish:
            frame_after = frame_before
        else:
            fresh = self._wait_for_fresh_frame(older_than_stamp_ns=frame_before.stamp_ns, deadline=deadline)
            frame_after = self._wait_until_settled(latest_frame=fresh, deadline=deadline)

        q_after = self._extract_q(frame_after)
        joint_delta = q_after - q_before
        joint_delta_l2 = float(np.linalg.norm(joint_delta))
        no_effect_by_abs = joint_delta_l2 < self.no_effect_l2
        effect_ratio = (joint_delta_l2 / cmd_delta_l2) if cmd_delta_l2 > 0.0 else 0.0
        no_effect_by_ratio = cmd_delta_l2 >= self.min_command_l2 and effect_ratio < self.no_effect_ratio
        no_effect = bool(skipped_publish or no_effect_by_abs or no_effect_by_ratio)

        return {
            "q_before": q_before.tolist(),
            "q_after": q_after.tolist(),
            "cmd_q": cmd_q.tolist(),
            "joint_delta": joint_delta.tolist(),
            "joint_delta_l2": joint_delta_l2,
            "cmd_delta_l2": cmd_delta_l2,
            "effect_ratio": float(effect_ratio),
            "no_effect": no_effect,
            "no_effect_reason": (
                "below_min_command"
                if skipped_publish
                else ("small_joint_delta" if no_effect_by_abs else ("small_effect_ratio" if no_effect_by_ratio else "none"))
            ),
            "skipped_publish": bool(skipped_publish),
            "frame_before_stamp_ns": int(frame_before.stamp_ns),
            "frame_after_stamp_ns": int(frame_after.stamp_ns),
            "timestamp_ns": int(time.time_ns()),
        }

    def close(self) -> None:
        close_fn = getattr(self.io, "close", None)
        if callable(close_fn):
            close_fn()
