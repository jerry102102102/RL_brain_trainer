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

    def execute_joint_target(
        self, joint_names: list[str], positions: np.ndarray, duration_s: float, result_timeout_s: float
    ) -> dict[str, Any]: ...


class ROS2JointRuntimeIO:
    """ROS2 I/O backend using FollowJointTrajectory action (primary) + topic fallback."""

    def __init__(
        self,
        trajectory_topic: str,
        joint_state_topic: str,
        joint_state_qos: str = "sensor_data",
        joint_state_qos_depth: int = 10,
        action_name: str = "/arm_controller/follow_joint_trajectory",
        use_action_primary: bool = True,
    ) -> None:
        try:
            import rclpy
            from trajectory_msgs.msg import JointTrajectory
            from sensor_msgs.msg import JointState
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient
            from action_msgs.msg import GoalStatus
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in ROS2 env
            raise RuntimeError("ROS2 dependencies are missing; source ROS2 and install msgs") from exc

        self._rclpy = rclpy
        self._JointTrajectory = JointTrajectory
        self._JointState = JointState
        self._FollowJointTrajectory = FollowJointTrajectory
        self._ActionClient = ActionClient
        self._GoalStatus = GoalStatus

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self._node = rclpy.create_node("v5_1_runtime_adapter")
        self._pub = self._node.create_publisher(JointTrajectory, trajectory_topic, 10)
        self._latest: JointStateFrame | None = None
        self._use_action_primary = bool(use_action_primary)
        self._action_client = ActionClient(self._node, FollowJointTrajectory, action_name) if self._use_action_primary else None

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

        sub_qos: Any = max(1, int(joint_state_qos_depth))
        if joint_state_qos == "sensor_data":
            try:
                from rclpy.qos import qos_profile_sensor_data

                sub_qos = qos_profile_sensor_data
            except Exception:
                sub_qos = max(1, int(joint_state_qos_depth))
        self._sub = self._node.create_subscription(JointState, joint_state_topic, _on_joint_state, sub_qos)

    def _build_joint_trajectory(self, joint_names: list[str], positions: np.ndarray, duration_s: float) -> Any:
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
        return msg

    def execute_joint_target(
        self, joint_names: list[str], positions: np.ndarray, duration_s: float, result_timeout_s: float
    ) -> dict[str, Any]:
        if self._action_client is None:
            return {
                "path": "topic_fallback",
                "accepted": True,
                "result_status": "unknown",
                "execution_ok": True,
                "fail_reason": "none",
                "action_status": None,
                "action_error_code": None,
            }

        if not self._action_client.wait_for_server(timeout_sec=max(0.01, float(result_timeout_s))):
            return {
                "path": "action",
                "accepted": False,
                "result_status": "rejected",
                "execution_ok": False,
                "fail_reason": "action_server_unavailable",
                "action_status": None,
                "action_error_code": None,
            }

        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory = self._build_joint_trajectory(joint_names, positions, duration_s)

        send_future = self._action_client.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=max(0.01, float(result_timeout_s)))
        if not send_future.done():
            return {
                "path": "action",
                "accepted": False,
                "result_status": "rejected",
                "execution_ok": False,
                "fail_reason": "goal_send_timeout",
                "action_status": None,
                "action_error_code": None,
            }

        goal_handle = send_future.result()
        accepted = bool(goal_handle is not None and getattr(goal_handle, "accepted", False))
        if not accepted:
            return {
                "path": "action",
                "accepted": False,
                "result_status": "rejected",
                "execution_ok": False,
                "fail_reason": "goal_rejected",
                "action_status": None,
                "action_error_code": None,
            }

        result_future = goal_handle.get_result_async()
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=max(0.01, float(result_timeout_s)))
        if not result_future.done():
            return {
                "path": "action",
                "accepted": True,
                "result_status": "fail",
                "execution_ok": False,
                "fail_reason": "action_result_timeout",
                "action_status": None,
                "action_error_code": None,
            }

        wrapped = result_future.result()
        status = int(getattr(wrapped, "status", 0))
        result = getattr(wrapped, "result", None)
        error_code = int(getattr(result, "error_code", 0)) if result is not None else 0

        status_ok = status == int(self._GoalStatus.STATUS_SUCCEEDED)
        code_ok = error_code == int(getattr(self._FollowJointTrajectory.Result, "SUCCESSFUL", 0))
        execution_ok = bool(status_ok and code_ok)

        fail_reason = "none"
        if not execution_ok:
            if not status_ok:
                fail_reason = f"action_status_{status}"
            elif not code_ok:
                fail_reason = f"action_error_code_{error_code}"

        return {
            "path": "action",
            "accepted": True,
            "result_status": "success" if execution_ok else "fail",
            "execution_ok": execution_ok,
            "fail_reason": fail_reason,
            "action_status": status,
            "action_error_code": error_code,
        }

    def publish_joint_target(self, joint_names: list[str], positions: np.ndarray, duration_s: float) -> None:
        msg = self._build_joint_trajectory(joint_names, positions, duration_s)
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
        initial_warmup_timeout_s: float = 2.5,
        initial_read_fallback_timeout_s: float = 2.0,
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
        self.initial_warmup_timeout_s = max(0.0, float(initial_warmup_timeout_s))
        self.initial_read_fallback_timeout_s = max(0.0, float(initial_read_fallback_timeout_s))
        self._has_initial_frame = False

        self._warmup_initial_frame()

    @classmethod
    def from_ros2(
        cls,
        joint_names: list[str],
        trajectory_topic: str = "/arm_controller/joint_trajectory",
        joint_state_topic: str = "/joint_states",
        command_duration_s: float = 0.2,
        settle_timeout_s: float = 2.5,
        initial_warmup_timeout_s: float = 2.5,
        initial_read_fallback_timeout_s: float = 2.0,
        joint_state_qos: str = "sensor_data",
        joint_state_qos_depth: int = 10,
        action_name: str = "/arm_controller/follow_joint_trajectory",
        use_action_primary: bool = True,
    ) -> "RuntimeROS2Adapter":
        return cls(
            io=ROS2JointRuntimeIO(
                trajectory_topic=trajectory_topic,
                joint_state_topic=joint_state_topic,
                joint_state_qos=joint_state_qos,
                joint_state_qos_depth=joint_state_qos_depth,
                action_name=action_name,
                use_action_primary=use_action_primary,
            ),
            joint_names=joint_names,
            command_duration_s=command_duration_s,
            settle_timeout_s=settle_timeout_s,
            initial_warmup_timeout_s=initial_warmup_timeout_s,
            initial_read_fallback_timeout_s=initial_read_fallback_timeout_s,
        )

    def _extract_q(self, frame: JointStateFrame) -> np.ndarray:
        idx = {name: i for i, name in enumerate(frame.names)}
        missing = [name for name in self.joint_names if name not in idx]
        if missing:
            raise ValueError(f"/joint_states missing joints: {missing}")
        return np.asarray([frame.position[idx[name]] for name in self.joint_names], dtype=float)

    def _warmup_initial_frame(self) -> None:
        if self.initial_warmup_timeout_s <= 0.0:
            return
        try:
            _ = self._read_frame_raw(timeout_s=self.initial_warmup_timeout_s)
            self._has_initial_frame = True
        except TimeoutError:
            # Bounded warmup is best-effort; read path below will classify/extend timeout.
            return

    def _read_frame_raw(self, timeout_s: float) -> JointStateFrame:
        frame = self.io.wait_for_joint_state(timeout_s=float(timeout_s))
        _ = self._extract_q(frame)
        return frame

    def _read_frame(self, timeout_s: float | None = None) -> JointStateFrame:
        primary_timeout_s = self.settle_timeout_s if timeout_s is None else float(timeout_s)

        if not self._has_initial_frame:
            try:
                frame = self._read_frame_raw(timeout_s=primary_timeout_s)
                self._has_initial_frame = True
                return frame
            except TimeoutError as exc:
                if self.initial_read_fallback_timeout_s > 0.0:
                    try:
                        frame = self._read_frame_raw(timeout_s=self.initial_read_fallback_timeout_s)
                        self._has_initial_frame = True
                        return frame
                    except TimeoutError as fallback_exc:
                        raise TimeoutError(
                            "joint_state_timeout_initial: "
                            f"primary={primary_timeout_s:.2f}s fallback={self.initial_read_fallback_timeout_s:.2f}s; "
                            f"err={fallback_exc}"
                        ) from fallback_exc
                raise TimeoutError(
                    f"joint_state_timeout_initial: primary={primary_timeout_s:.2f}s err={exc}"
                ) from exc

        try:
            return self._read_frame_raw(timeout_s=primary_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"joint_state_timeout_step: timeout={primary_timeout_s:.2f}s err={exc}") from exc

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
        raise TimeoutError("joint_state_timeout_step: timed out waiting for fresh joint state frame")

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
        transport = {
            "path": "topic_fallback",
            "accepted": True,
            "result_status": "success",
            "execution_ok": None,
            "fail_reason": "none",
            "action_status": None,
            "action_error_code": None,
        }

        if skipped_publish:
            transport.update(
                {
                    "accepted": False,
                    "result_status": "fail",
                    "execution_ok": False,
                    "fail_reason": "below_min_command",
                }
            )
        else:
            exec_fn = getattr(self.io, "execute_joint_target", None)
            if callable(exec_fn):
                transport = exec_fn(self.joint_names, cmd_q, self.command_duration_s, self.settle_timeout_s)
                if str(transport.get("path")) == "topic_fallback":
                    self.io.publish_joint_target(self.joint_names, cmd_q, self.command_duration_s)
            else:
                self.io.publish_joint_target(self.joint_names, cmd_q, self.command_duration_s)

        deadline = time.monotonic() + self.settle_timeout_s
        if skipped_publish or not bool(transport.get("accepted", True)):
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

        no_effect_reason = (
            "below_min_command"
            if skipped_publish
            else ("small_joint_delta" if no_effect_by_abs else ("small_effect_ratio" if no_effect_by_ratio else "none"))
        )
        transport_execution_ok = transport.get("execution_ok")
        execution_ok = bool((not no_effect) if transport_execution_ok is None else transport_execution_ok)
        fail_reason = str(transport.get("fail_reason", "none" if execution_ok else no_effect_reason))
        result_status = str(transport.get("result_status", "success" if execution_ok else "fail"))
        accepted = bool(transport.get("accepted", not skipped_publish))

        if not execution_ok and fail_reason == "none":
            fail_reason = no_effect_reason if no_effect else "execution_failed"

        return {
            "q_before": q_before.tolist(),
            "q_after": q_after.tolist(),
            "cmd_q": cmd_q.tolist(),
            "joint_delta": joint_delta.tolist(),
            "joint_delta_l2": joint_delta_l2,
            "cmd_delta_l2": cmd_delta_l2,
            "effect_ratio": float(effect_ratio),
            "no_effect": no_effect,
            "no_effect_reason": no_effect_reason,
            "skipped_publish": bool(skipped_publish),
            "accepted": accepted,
            "result_status": result_status,
            "execution_ok": execution_ok,
            "fail_reason": fail_reason,
            "command_path": str(transport.get("path", "topic_fallback")),
            "action_status": transport.get("action_status"),
            "action_error_code": transport.get("action_error_code"),
            "frame_before_stamp_ns": int(frame_before.stamp_ns),
            "frame_after_stamp_ns": int(frame_after.stamp_ns),
            "timestamp_ns": int(time.time_ns()),
        }

    def close(self) -> None:
        close_fn = getattr(self.io, "close", None)
        if callable(close_fn):
            close_fn()
