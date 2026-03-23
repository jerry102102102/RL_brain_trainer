"""V5 Task-1 EE pose reaching trainer.

Task split alignment:
- L1: target EE pose provider (high/safe pose by default)
- L2: learnable policy layer (with replay/update/checkpoint support)
- L3: safety + execution interface (bootstrap simulator or Gazebo runtime path)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np

RewardMode = Literal["no_shaping", "heuristic", "pbrs"]
RuntimeBackend = Literal["bootstrap", "gazebo"]


@lru_cache(maxsize=1)
def _load_legacy_fk_solver() -> Callable[[np.ndarray], np.ndarray] | None:
    """Load FK solver from legacy ENPM662 controller code by absolute file path."""
    repo_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "external" / "ENPM662_Group4_FinalProject").exists():
            repo_root = parent
            break
    if repo_root is None:
        return None

    kin_path = (
        repo_root
        / "external"
        / "ENPM662_Group4_FinalProject"
        / "src"
        / "kitchen_robot_controller"
        / "kitchen_robot_controller"
        / "kinematics.py"
    )
    if not kin_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("legacy_kitchen_robot_kinematics", str(kin_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Required for dataclass/type introspection during module import.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fk = getattr(module, "fk_ur", None)
        return fk if callable(fk) else None
    except Exception:
        return None


@dataclass(frozen=True)
class Task1Config:
    n_joints: int = 6
    dt: float = 0.1
    max_steps: int = 40
    max_delta_q: float = 0.05
    success_pos_tol: float = 0.03
    safe_z_min: float = 0.20
    safety_margin_min: float = 0.0
    step_penalty: float = -0.01
    success_bonus: float = 2.0
    fail_penalty: float = -1.0
    gamma: float = 0.99


@dataclass
class Task1State:
    q: np.ndarray
    dq: np.ndarray
    target_pose_xyz: np.ndarray
    step: int
    max_steps: int
    safe_z_min: float
    ee_proxy_xyz: np.ndarray | None = None

    @property
    def ee_pos(self) -> np.ndarray:
        if self.ee_proxy_xyz is not None:
            return self.ee_proxy_xyz
        # Bootstrap fallback mapping: q[:3] maps to EE xyz in simulator coordinates.
        return self.q[:3]


@dataclass(frozen=True)
class Task1Observation:
    q: np.ndarray
    dq: np.ndarray
    delta_p: np.ndarray
    d_pos: float
    t_remain: float
    z_margin: float

    def to_dict(self) -> dict[str, object]:
        return {
            "q": self.q.tolist(),
            "dq": self.dq.tolist(),
            "delta_p": self.delta_p.tolist(),
            "d_pos": float(self.d_pos),
            "t_remain": float(self.t_remain),
            "z_margin": float(self.z_margin),
        }


@dataclass(frozen=True)
class L2Action:
    delta_q_raw: np.ndarray


@dataclass(frozen=True)
class MacroDecision:
    decision_id: str
    state_version: int
    ttl_steps: int
    target_q: np.ndarray
    seed_delta_q: np.ndarray


@dataclass(frozen=True)
class L3ExecutionResult:
    accepted: bool
    q_next: np.ndarray
    dq_next: np.ndarray
    safety_violation: float
    ee_proxy_xyz: np.ndarray | None = None
    logs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayTransition:
    d_pos_prev: float
    d_pos_next: float
    reward: float


class L2PolicyContract(Protocol):
    def decide_action(self, obs: Task1Observation) -> L2Action: ...


class L3ExecutorContract(Protocol):
    def execute_with_safety(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult: ...


@dataclass
class HighPoseTargetProvider:
    """L1 target provider with default high-Z pose to avoid collisions."""

    target_xyz: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.0, 0.35], dtype=float))

    def get_target_pose(self, episode_index: int) -> np.ndarray:
        # Deterministic for bootstrap reproducibility.
        return self.target_xyz.copy()


@dataclass
class LearnableL2Policy:
    """Minimal learnable L2: adaptive proportional gain updated from replay."""

    gain: float = 0.7
    gain_min: float = 0.1
    gain_max: float = 1.6

    def decide_action(self, obs: Task1Observation) -> L2Action:
        cmd_xyz = self.gain * obs.delta_p
        raw = np.zeros_like(obs.q)
        raw[:3] = cmd_xyz
        return L2Action(delta_q_raw=raw)

    def update_from_replay(self, replay: list[ReplayTransition], lr: float = 0.05) -> None:
        if not replay:
            return
        progress = [r.d_pos_prev - r.d_pos_next for r in replay]
        mean_progress = float(np.mean(progress))
        mean_reward = float(np.mean([r.reward for r in replay]))
        direction = 1.0 if (mean_progress > 1e-4 and mean_reward > -0.2) else -1.0
        self.gain = float(np.clip(self.gain + direction * lr, self.gain_min, self.gain_max))

    def to_checkpoint(self) -> dict[str, float]:
        return {"gain": float(self.gain), "gain_min": float(self.gain_min), "gain_max": float(self.gain_max)}


@dataclass
class SafetyConstrainedL3Executor:
    """Bootstrap L3 path: limit command, apply safety checks, then synthetic state update."""

    q_min: np.ndarray = field(default_factory=lambda: np.array([-0.70, -1.5, 0.0, -2.0, -2.0, -2.0, -2.0], dtype=float))
    q_max: np.ndarray = field(default_factory=lambda: np.array([0.70, 1.5, 1.2, 2.0, 2.0, 2.0, 2.0], dtype=float))
    max_dq_per_step: float = 0.05

    def execute_with_safety(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult:
        limited_cmd = np.clip(delta_q_cmd, -self.max_dq_per_step, self.max_dq_per_step)
        q_candidate = np.clip(state.q + limited_cmd, self.q_min[: state.q.size], self.q_max[: state.q.size])
        dq_candidate = (q_candidate - state.q).copy()

        ee_z_next = float(q_candidate[2])
        if ee_z_next < state.safe_z_min:
            return L3ExecutionResult(
                accepted=False,
                q_next=state.q.copy(),
                dq_next=np.zeros_like(state.dq),
                safety_violation=float(state.safe_z_min - ee_z_next),
                ee_proxy_xyz=None,
                logs=(
                    "L3_CHECK:z_under_safe_min",
                    f"safe_z_min={state.safe_z_min:.3f}",
                    f"ee_z_next={ee_z_next:.3f}",
                    "L3_EXEC:rejected",
                ),
            )

        return L3ExecutionResult(
            accepted=True,
            q_next=q_candidate,
            dq_next=dq_candidate,
            safety_violation=0.0,
            ee_proxy_xyz=None,
            logs=("L3_CHECK:ok", "L3_EXEC:accepted", "L3_EXEC:path=bootstrap"),
        )


@dataclass
class GazeboRuntimeL3Executor:
    """Gazebo-backed runtime executor.

    Fail-fast behavior:
    - require /joint_states
    - require /arm_controller/joint_trajectory publisher path
    - require at least one runtime joint state sample before training starts
    - require real EE pose source (topic or TF) unless explicit fallback is enabled
    """

    max_dq_per_step: float = 0.05
    command_topic: str = "/arm_controller/joint_trajectory"
    joint_state_topic: str = "/joint_states"
    ee_pose_topic: str = "/ee_pose"
    joint_state_wait_sec: float = 3.0
    action_timeout_sec: float = 1.5
    action_server_timeout_sec: float = 3.0
    l3_exec_mode: Literal["action", "topic", "hybrid"] = "action"
    macro_kickoff_action: bool = False
    action_name: str = "/arm_controller/follow_joint_trajectory"
    action_min_motion_tol: float = 1e-4
    reset_timeout_sec: float = 4.0
    scene_reset_cmd: str | None = None
    reset_position_tol: float = 0.03
    allow_ee_fallback: bool = False
    use_tf_ee_pose: bool = True
    ee_base_frame: str = "base_link"
    ee_tip_frame: str = "tool0"
    verbose_debug: bool = False
    q_min: np.ndarray = field(default_factory=lambda: np.array([-0.70, -1.5, 0.0, -2.0, -2.0, -2.0, -2.0], dtype=float))
    q_max: np.ndarray = field(default_factory=lambda: np.array([0.70, 1.5, 1.2, 2.0, 2.0, 2.0, 2.0], dtype=float))

    def __post_init__(self) -> None:
        try:
            import rclpy
            from control_msgs.action import FollowJointTrajectory
            from geometry_msgs.msg import PoseStamped
            from rclpy.action import ActionClient
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except Exception as exc:  # pragma: no cover - depends on runtime ROS install
            raise RuntimeError(
                "backend=gazebo requires ROS2 Python packages "
                "(rclpy, sensor_msgs, trajectory_msgs, control_msgs)"
            ) from exc

        self._rclpy = rclpy
        self._JointState = JointState
        self._PoseStamped = PoseStamped
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._FollowJointTrajectory = FollowJointTrajectory
        self._ActionClient = ActionClient

        rclpy.init(args=None)
        self._node = rclpy.create_node("task1_gazebo_runtime_executor")
        self._latest_joint_state: JointState | None = None
        self._latest_ee_pose_xyz: np.ndarray | None = None
        self._latest_joint_stamp_ns: int | None = None

        self._node.create_subscription(self._JointState, self.joint_state_topic, self._on_joint_state, 10)
        self._node.create_subscription(self._PoseStamped, self.ee_pose_topic, self._on_ee_pose, 10)
        self._traj_pub = self._node.create_publisher(self._JointTrajectory, self.command_topic, 10)
        self._traj_action_client = self._ActionClient(self._node, self._FollowJointTrajectory, self.action_name)

        self._tf_buffer = None
        self._tf_listener = None
        if self.use_tf_ee_pose:
            try:
                from tf2_ros import Buffer, TransformListener

                self._tf_buffer = Buffer()
                self._tf_listener = TransformListener(self._tf_buffer, self._node, spin_thread=False)
            except Exception:
                self._tf_buffer = None
                self._tf_listener = None

        self._fk_solver = _load_legacy_fk_solver()
        self._legacy_fk_joint_order = (
            "Rack_joint",
            "robot_base_joint",
            "shoulder1_joint",
            "shoulder2_joint",
            "wr1_joint",
            "wr2_joint",
            "wr3_joint",
        )

        self._spin_until_discovery_or_fail()
        if self.l3_exec_mode in {"action", "hybrid"} and self.macro_kickoff_action:
            self._wait_for_action_server_or_fail()
        self._wait_for_initial_joint_state_or_fail()

    def close(self) -> None:
        if getattr(self, "_node", None) is not None:
            self._node.destroy_node()
        if getattr(self, "_rclpy", None) is not None and self._rclpy.ok():
            self._rclpy.shutdown()

    def _spin_once(self, timeout_sec: float = 0.1) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def _spin_until_discovery_or_fail(self) -> None:
        deadline = time.time() + float(self.joint_state_wait_sec)
        while time.time() < deadline:
            self._spin_once(0.1)
            topics = {name: types for name, types in self._node.get_topic_names_and_types()}
            has_joint_state = self.joint_state_topic in topics
            has_cmd_topic = self.command_topic in topics
            if self.l3_exec_mode == "action":
                if has_joint_state:
                    return
            elif has_joint_state and has_cmd_topic:
                return
        if self.l3_exec_mode == "action":
            raise RuntimeError(
                "backend=gazebo fail-fast: required topics missing. "
                f"need {self.joint_state_topic} for l3_exec_mode=action"
            )
        raise RuntimeError(
            "backend=gazebo fail-fast: required topics missing. "
            f"need {self.joint_state_topic} and {self.command_topic}"
        )

    def _wait_for_action_server_or_fail(self) -> None:
        if self._traj_action_client.wait_for_server(timeout_sec=float(self.action_server_timeout_sec)):
            return
        raise RuntimeError(
            "backend=gazebo fail-fast: FollowJointTrajectory action server unavailable "
            f"name={self.action_name} timeout={self.action_server_timeout_sec:.2f}s"
        )

    def _wait_for_initial_joint_state_or_fail(self) -> None:
        deadline = time.time() + float(self.joint_state_wait_sec)
        while time.time() < deadline:
            self._spin_once(0.1)
            if self._latest_joint_state is not None:
                return
        raise RuntimeError(
            "backend=gazebo fail-fast: no /joint_states samples received within "
            f"{self.joint_state_wait_sec:.1f}s"
        )

    def _on_joint_state(self, msg) -> None:
        self._latest_joint_state = msg
        self._latest_joint_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_ee_pose(self, msg) -> None:
        self._latest_ee_pose_xyz = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)

    def _lookup_ee_pose_from_tf(self) -> np.ndarray | None:
        if self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(self.ee_base_frame, self.ee_tip_frame, self._rclpy.time.Time())
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z], dtype=float)
        except Exception:
            return None

    def _lookup_ee_pose_from_fk_joint_state(self) -> np.ndarray | None:
        if self._fk_solver is None:
            return None
        if self._latest_joint_state is None:
            if self.verbose_debug:
                print("[gazebo-debug] fk_skip: no joint_state message cached yet")
            return None
        names = list(getattr(self._latest_joint_state, "name", []) or [])
        positions = list(getattr(self._latest_joint_state, "position", []) or [])
        if not names:
            if self.verbose_debug:
                print("[gazebo-debug] fk_skip: joint_state has no joint names")
            return None
        if not positions:
            if self.verbose_debug:
                print("[gazebo-debug] fk_skip: joint_state has no joint positions")
            return None

        index_by_name = {name: idx for idx, name in enumerate(names)}
        missing = [jn for jn in self._legacy_fk_joint_order if jn not in index_by_name]
        if missing:
            if self.verbose_debug:
                print(
                    "[gazebo-debug] fk_skip: required FK joint names missing "
                    f"missing={missing} available={names}"
                )
            return None

        ordered_indices = [index_by_name[jn] for jn in self._legacy_fk_joint_order]
        max_required_idx = max(ordered_indices) if ordered_indices else -1
        if len(positions) <= max_required_idx:
            if self.verbose_debug:
                print(
                    "[gazebo-debug] fk_skip: joint_state position count too small for FK mapping "
                    f"positions={len(positions)} max_required_index={max_required_idx}"
                )
            return None

        try:
            q_fk = np.array([positions[idx] for idx in ordered_indices], dtype=float)
            t_ee = self._fk_solver(q_fk)
            if t_ee is None:
                if self.verbose_debug:
                    print("[gazebo-debug] fk_skip: fk solver returned None")
                return None
            pos = np.asarray(t_ee[:3, 3], dtype=float)
            return pos.copy()
        except Exception as exc:
            if self.verbose_debug:
                print(f"[gazebo-debug] fk_skip: fk solver exception={exc!r}")
            return None

    def _resolve_ee_pose_or_fail(self, q: np.ndarray) -> tuple[np.ndarray, str]:
        if self._latest_ee_pose_xyz is not None:
            return self._latest_ee_pose_xyz.copy(), "ee_pose_topic"
        tf_pose = self._lookup_ee_pose_from_tf()
        if tf_pose is not None:
            return tf_pose, "tf_lookup"
        fk_pose = self._lookup_ee_pose_from_fk_joint_state()
        if fk_pose is not None:
            return fk_pose, "fk_joint_state"
        if self.allow_ee_fallback:
            return q[:3].copy(), "joint_state_first3_fallback"
        raise RuntimeError(
            "backend=gazebo fail-fast: no EE pose source available "
            f"(topic={self.ee_pose_topic}, tf={self.ee_base_frame}->{self.ee_tip_frame}, fk=legacy_controller_joint_state). "
            "Use --allow-ee-fallback only for diagnostic runs."
        )

    def ee_source_name(self) -> str:
        if self._latest_ee_pose_xyz is not None:
            return "ee_pose_topic"
        if self._lookup_ee_pose_from_tf() is not None:
            return "tf_lookup"
        if self._lookup_ee_pose_from_fk_joint_state() is not None:
            return "fk_joint_state"
        if self.allow_ee_fallback:
            return "joint_state_first3_fallback"
        return "unavailable"

    def runtime_joint_names(self, n_joints: int) -> tuple[str, ...]:
        if self._latest_joint_state is None:
            return tuple(f"joint_{idx}" for idx in range(int(n_joints)))
        names = tuple(self._latest_joint_state.name[:n_joints]) if self._latest_joint_state.name else tuple()
        if names:
            return names
        return tuple(f"joint_{idx}" for idx in range(int(n_joints)))

    def read_runtime_state(self, n_joints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        self._spin_once(0.05)
        if self._latest_joint_state is None:
            raise RuntimeError("backend=gazebo fail-fast: joint state stream dropped")

        pos = np.asarray(self._latest_joint_state.position, dtype=float)
        vel = np.asarray(self._latest_joint_state.velocity, dtype=float) if self._latest_joint_state.velocity else np.zeros_like(pos)
        if pos.size < n_joints:
            raise RuntimeError(
                "backend=gazebo fail-fast: /joint_states has fewer joints than expected "
                f"({pos.size} < {n_joints})"
            )
        q = pos[:n_joints].copy()
        dq = vel[:n_joints].copy() if vel.size >= n_joints else np.zeros(n_joints, dtype=float)

        ee_proxy, ee_source = self._resolve_ee_pose_or_fail(q)
        if self.verbose_debug:
            fk_ok = self._lookup_ee_pose_from_fk_joint_state() is not None
            print(
                f"[gazebo-debug] runtime_state ee_source={ee_source} fk_success={fk_ok} "
                f"ee_pos={ee_proxy.tolist()} q0_3={q[:3].tolist()}"
            )
        return q, dq, ee_proxy

    def execute_with_safety(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult:
        q_runtime, _, ee_runtime = self.read_runtime_state(n_joints=state.q.size)
        limited_cmd = np.clip(delta_q_cmd, -self.max_dq_per_step, self.max_dq_per_step)
        q_target = np.clip(q_runtime + limited_cmd, self.q_min[: q_runtime.size], self.q_max[: q_runtime.size])

        candidate_ee_z = float(ee_runtime[2]) if ee_runtime is not None else float(q_runtime[2])
        if candidate_ee_z < state.safe_z_min:
            diag = (
                f"unsafe reject: reason=z_under_safe_min safe_z_min={state.safe_z_min:.3f} "
                f"candidate_ee_z={candidate_ee_z:.3f} margin={candidate_ee_z - state.safe_z_min:.3f} "
                f"ee_source={self.ee_source_name()}"
            )
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=float(state.safe_z_min - candidate_ee_z),
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=("L3_CHECK:z_under_safe_min", f"safe_z_min={state.safe_z_min:.3f}", diag, "L3_EXEC:rejected", "L3_EXEC:path=gazebo"),
            )

        if self.l3_exec_mode in {"topic", "hybrid"}:
            return self._execute_via_topic(state=state, q_runtime=q_runtime, q_target=q_target, ee_runtime=ee_runtime)
        return self._execute_via_action(state=state, q_runtime=q_runtime, q_target=q_target, ee_runtime=ee_runtime)

    def execute_macro_kickoff(self, state: Task1State, delta_q_cmd: np.ndarray) -> L3ExecutionResult:
        if self.l3_exec_mode != "hybrid" or not self.macro_kickoff_action:
            return self.execute_with_safety(state, delta_q_cmd)

        q_runtime, _, ee_runtime = self.read_runtime_state(n_joints=state.q.size)
        limited_cmd = np.clip(delta_q_cmd, -self.max_dq_per_step, self.max_dq_per_step)
        q_target = np.clip(q_runtime + limited_cmd, self.q_min[: q_runtime.size], self.q_max[: q_runtime.size])
        return self._execute_via_action(state=state, q_runtime=q_runtime, q_target=q_target, ee_runtime=ee_runtime)

    def _finalize_execution_result(
        self,
        *,
        state: Task1State,
        q_runtime: np.ndarray,
        ee_runtime: np.ndarray | None,
        base_logs: list[str],
        status_ok: bool,
    ) -> L3ExecutionResult:
        q_next, dq_next, ee_proxy = self.read_runtime_state(n_joints=state.q.size)
        q_delta = q_next - q_runtime
        ee_before = ee_runtime if ee_runtime is not None else q_runtime[:3]
        ee_after = ee_proxy if ee_proxy is not None else q_next[:3]
        ee_delta = ee_after - ee_before
        delta_max = float(np.max(np.abs(q_delta)))
        ee_delta_norm = float(np.linalg.norm(ee_delta))
        ee_z = float(ee_after[2])
        safety_violation = float(max(0.0, state.safe_z_min - ee_z))
        moved = delta_max >= float(self.action_min_motion_tol)
        accepted = bool(status_ok and moved)

        logs = list(base_logs)
        logs.extend(
            [
                f"post_action_joint_delta_max={delta_max:.6f}",
                f"post_action_ee_delta_norm={ee_delta_norm:.6f}",
                f"z_margin={ee_z - state.safe_z_min:.3f}",
                f"EE_SOURCE:{self.ee_source_name()}",
            ]
        )
        if not moved:
            logs.append(f"L3_EXEC:no_motion min_motion_tol={self.action_min_motion_tol}")
        logs.append("L3_EXEC:accepted" if accepted else "L3_EXEC:rejected")
        return L3ExecutionResult(
            accepted=accepted,
            q_next=q_next,
            dq_next=dq_next,
            safety_violation=safety_violation,
            ee_proxy_xyz=ee_proxy,
            logs=tuple(logs),
        )

    def _execute_via_topic(
        self,
        *,
        state: Task1State,
        q_runtime: np.ndarray,
        q_target: np.ndarray,
        ee_runtime: np.ndarray | None,
    ) -> L3ExecutionResult:
        msg = self._JointTrajectory()
        msg.joint_names = list(self._latest_joint_state.name[: state.q.size])
        msg.header.stamp = self._node.get_clock().now().to_msg()
        point = self._JointTrajectoryPoint()
        point.positions = q_target.tolist()
        point.time_from_start.sec = max(1, int(np.ceil(state.max_steps * 0.0 + 1)))
        msg.points = [point]
        self._traj_pub.publish(msg)

        stamp_before = self._latest_joint_stamp_ns
        deadline = time.time() + float(self.action_timeout_sec)
        while time.time() < deadline:
            self._spin_once(0.05)
            if self._latest_joint_stamp_ns is not None and self._latest_joint_stamp_ns != stamp_before:
                return self._finalize_execution_result(
                    state=state,
                    q_runtime=q_runtime,
                    ee_runtime=ee_runtime,
                    base_logs=["L3_CHECK:ok", f"safe_z_min={state.safe_z_min:.3f}", "L3_EXEC:path=gazebo", "L3_EXEC:mode=topic"],
                    status_ok=True,
                )

        return L3ExecutionResult(
            accepted=False,
            q_next=q_runtime,
            dq_next=np.zeros_like(q_runtime),
            safety_violation=0.0,
            ee_proxy_xyz=state.ee_proxy_xyz,
            logs=("L3_EXEC:timeout_or_no_motion", f"min_motion_tol={self.action_min_motion_tol}", "L3_EXEC:path=gazebo", "L3_EXEC:mode=topic"),
        )

    def _execute_via_action(
        self,
        *,
        state: Task1State,
        q_runtime: np.ndarray,
        q_target: np.ndarray,
        ee_runtime: np.ndarray | None,
    ) -> L3ExecutionResult:
        if not self._traj_action_client.wait_for_server(timeout_sec=float(self.action_server_timeout_sec)):
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=0.0,
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=(
                    "L3_EXEC:rejected",
                    "L3_EXEC:path=gazebo",
                    "L3_EXEC:mode=action",
                    f"action_server_timeout={self.action_server_timeout_sec:.2f}s",
                    f"action_name={self.action_name}",
                ),
            )

        goal = self._FollowJointTrajectory.Goal()
        trajectory = self._JointTrajectory()
        trajectory.joint_names = list(self._latest_joint_state.name[: state.q.size])
        trajectory.header.stamp = self._node.get_clock().now().to_msg()
        point = self._JointTrajectoryPoint()
        point.positions = q_target.tolist()
        # Give controller enough execution window for action result feedback.
        point.time_from_start.sec = max(2, int(np.ceil(self.action_timeout_sec)))
        trajectory.points = [point]
        goal.trajectory = trajectory

        send_log = (
            f"action_goal_sent joints={trajectory.joint_names} "
            f"positions0_3={point.positions[:3]} "
            f"time_from_start_sec={point.time_from_start.sec}"
        )

        send_future = self._traj_action_client.send_goal_async(goal)
        self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=float(self.action_timeout_sec))
        if not send_future.done():
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=0.0,
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=(send_log, "L3_EXEC:rejected", "L3_EXEC:mode=action", f"action_goal_response_timeout={self.action_timeout_sec:.2f}s"),
            )

        goal_handle = send_future.result()
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=0.0,
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=(send_log, "L3_EXEC:rejected", "L3_EXEC:mode=action", "action_goal_response=rejected"),
            )

        result_future = goal_handle.get_result_async()
        result_wait_timeout = max(float(self.action_timeout_sec), float(point.time_from_start.sec) + 2.0)
        try:
            self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=result_wait_timeout)
        except Exception as exc:
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=0.0,
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=(send_log, "L3_EXEC:rejected", "L3_EXEC:mode=action", f"action_wait_exception={exc!r}"),
            )
        if not result_future.done():
            return L3ExecutionResult(
                accepted=False,
                q_next=q_runtime,
                dq_next=np.zeros_like(q_runtime),
                safety_violation=0.0,
                ee_proxy_xyz=ee_runtime.copy() if ee_runtime is not None else state.ee_proxy_xyz,
                logs=(send_log, "L3_EXEC:rejected", "L3_EXEC:mode=action", f"action_result_timeout={result_wait_timeout:.2f}s"),
            )

        result_msg = result_future.result()
        status = int(getattr(result_msg, "status", -1))
        status_map = {
            0: "UNKNOWN",
            1: "ACCEPTED",
            2: "EXECUTING",
            3: "CANCELING",
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "ABORTED",
        }
        status_name = status_map.get(status, f"UNMAPPED_{status}")
        status_ok = status == 4
        return self._finalize_execution_result(
            state=state,
            q_runtime=q_runtime,
            ee_runtime=ee_runtime,
            base_logs=[
                "L3_CHECK:ok",
                f"safe_z_min={state.safe_z_min:.3f}",
                "L3_EXEC:path=gazebo",
                "L3_EXEC:mode=action",
                send_log,
                "action_goal_response=accepted",
                f"action_result_status={status}",
                f"action_result_status_name={status_name}",
            ],
            status_ok=status_ok,
        )

    def reset_arm_to_initial(
        self,
        *,
        n_joints: int,
        timeout_sec: float,
        target_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        q_runtime, _, _ = self.read_runtime_state(n_joints=n_joints)
        q_last = q_runtime.copy()
        q_goal = np.zeros(n_joints, dtype=float) if target_q is None else np.asarray(target_q, dtype=float).copy()
        q_goal = np.clip(q_goal, self.q_min[:n_joints], self.q_max[:n_joints])

        if self._traj_action_client.wait_for_server(timeout_sec=float(timeout_sec)):
            goal = self._FollowJointTrajectory.Goal()
            trajectory = self._JointTrajectory()
            trajectory.joint_names = list(self._latest_joint_state.name[:n_joints])
            trajectory.header.stamp = self._node.get_clock().now().to_msg()
            point = self._JointTrajectoryPoint()
            point.positions = q_goal.tolist()
            point.time_from_start.sec = max(1, int(np.ceil(timeout_sec)))
            trajectory.points = [point]
            goal.trajectory = trajectory
            send_future = self._traj_action_client.send_goal_async(goal)
            self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=float(timeout_sec))
        else:
            msg = self._JointTrajectory()
            msg.joint_names = list(self._latest_joint_state.name[:n_joints])
            msg.header.stamp = self._node.get_clock().now().to_msg()
            point = self._JointTrajectoryPoint()
            point.positions = q_goal.tolist()
            point.time_from_start.sec = max(1, int(np.ceil(timeout_sec)))
            msg.points = [point]
            self._traj_pub.publish(msg)

        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            self._spin_once(0.05)
            q_now, dq_now, ee_proxy_now = self.read_runtime_state(n_joints=n_joints)
            q_last = q_now
            if float(np.max(np.abs(q_now - q_goal))) <= float(self.reset_position_tol):
                return q_now, dq_now, ee_proxy_now

        raise RuntimeError(
            "backend=gazebo fail-fast: arm reset did not converge within "
            f"{timeout_sec:.2f}s (max_abs_error={float(np.max(np.abs(q_last - q_goal))):.4f})"
        )

    def reset_episode(self, *, n_joints: int, timeout_sec: float | None = None) -> dict[str, object]:
        start_ts = time.perf_counter()
        effective_timeout = float(self.reset_timeout_sec if timeout_sec is None else timeout_sec)
        reset_meta: dict[str, object] = {
            "applied": True,
            "success": False,
            "duration_ms": 0.0,
            "initial_state": None,
            "error": None,
        }
        try:
            if self.scene_reset_cmd:
                scene_proc = subprocess.run(
                    self.scene_reset_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    check=False,
                )
                if scene_proc.returncode != 0:
                    stderr = (scene_proc.stderr or "").strip()
                    raise RuntimeError(
                        "backend=gazebo fail-fast: scene reset command failed "
                        f"(code={scene_proc.returncode}): {stderr or '<empty stderr>'}"
                    )

            q0, dq0, ee_proxy0 = self.reset_arm_to_initial(n_joints=n_joints, timeout_sec=effective_timeout)
            reset_meta["initial_state"] = {
                "q": q0.tolist(),
                "dq": dq0.tolist(),
                "ee_pose": ee_proxy0.tolist() if ee_proxy0 is not None else None,
                "ee_source": self.ee_source_name(),
            }
            reset_meta["success"] = True
            return reset_meta
        except Exception as exc:
            reset_meta["error"] = str(exc)
            err = RuntimeError(f"backend=gazebo fail-fast: reset_episode failed: {exc}")
            setattr(err, "reset_meta", reset_meta)
            raise err from exc
        finally:
            reset_meta["duration_ms"] = float((time.perf_counter() - start_ts) * 1000.0)


def build_task1_observation(state: Task1State) -> Task1Observation:
    delta_p = state.target_pose_xyz - state.ee_pos
    d_pos = float(np.linalg.norm(delta_p))
    t_remain = max(0.0, float(state.max_steps - state.step) / float(state.max_steps))
    z_margin = float(state.ee_pos[2] - state.safe_z_min)
    return Task1Observation(
        q=state.q.copy(),
        dq=state.dq.copy(),
        delta_p=delta_p,
        d_pos=d_pos,
        t_remain=t_remain,
        z_margin=z_margin,
    )


def adapt_action_delta_q(action: L2Action, *, n_joints: int, max_delta_q: float) -> np.ndarray:
    raw = np.asarray(action.delta_q_raw, dtype=float)
    if raw.shape != (n_joints,):
        raise ValueError(f"delta_q_raw shape mismatch: expected {(n_joints,)}, got {raw.shape}")
    return np.clip(raw, -max_delta_q, max_delta_q)


def compose_task1_reward(
    *,
    mode: RewardMode,
    obs_prev: Task1Observation,
    obs_next: Task1Observation,
    delta_q_cmd: np.ndarray,
    safety_violation: float,
    done: bool,
    success: bool,
    cfg: Task1Config,
) -> float:
    smooth_penalty = -0.05 * float(np.linalg.norm(delta_q_cmd))
    safety_penalty = -2.0 * float(max(0.0, safety_violation))

    if mode == "no_shaping":
        reward = cfg.step_penalty
    elif mode == "heuristic":
        reward = -obs_next.d_pos + smooth_penalty + safety_penalty
    elif mode == "pbrs":
        phi_prev = -obs_prev.d_pos
        phi_next = -obs_next.d_pos
        reward = cfg.gamma * phi_next - phi_prev + smooth_penalty + safety_penalty
    else:
        raise ValueError(f"Unsupported reward mode: {mode}")

    if done:
        reward += cfg.success_bonus if success else cfg.fail_penalty
    return float(reward)


def check_done_success(state: Task1State, obs: Task1Observation, *, safety_violation: float, cfg: Task1Config) -> tuple[bool, bool, str | None]:
    success = obs.d_pos <= cfg.success_pos_tol and obs.z_margin >= cfg.safety_margin_min
    timeout = state.step >= state.max_steps
    unsafe = safety_violation > 0.0
    done = bool(success or timeout or unsafe)
    if success:
        return done, True, "success"
    if unsafe:
        return done, False, "unsafe"
    if timeout:
        return done, False, "timeout"
    return done, False, None


def run_task1_episode(
    *,
    episode_index: int,
    reward_mode: RewardMode,
    cfg: Task1Config,
    l1_provider: HighPoseTargetProvider,
    l2_policy: L2PolicyContract,
    l3_executor: L3ExecutorContract,
    initial_q: np.ndarray | None = None,
    initial_dq: np.ndarray | None = None,
    initial_ee_proxy_xyz: np.ndarray | None = None,
    verbose_debug: bool = False,
    episode_debug_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    state = Task1State(
        q=initial_q.copy() if initial_q is not None else np.array([0.0, 0.0, max(cfg.safe_z_min + 0.02, 0.22), 0.0, 0.0, 0.0], dtype=float),
        dq=initial_dq.copy() if initial_dq is not None else np.zeros(cfg.n_joints, dtype=float),
        target_pose_xyz=l1_provider.get_target_pose(episode_index=episode_index),
        step=0,
        max_steps=cfg.max_steps,
        safe_z_min=cfg.safe_z_min,
        ee_proxy_xyz=initial_ee_proxy_xyz.copy() if initial_ee_proxy_xyz is not None else None,
    )

    traj: list[dict[str, object]] = []
    replay: list[ReplayTransition] = []
    total_reward = 0.0
    done = False
    if verbose_debug:
        print(
            f"[episode-start] ep={episode_index} "
            f"backend={(episode_debug_meta or {}).get('backend', 'unknown')} "
            f"safe_z_min={cfg.safe_z_min:.3f} "
            f"ee_source={(episode_debug_meta or {}).get('ee_source', 'unknown')} "
            f"reset={(episode_debug_meta or {}).get('reset_summary', 'n/a')}"
        )
    success = False
    term_reason: str | None = None
    decision_ttl_steps = int((episode_debug_meta or {}).get("macro_ttl_steps", 4))
    state_version = 0
    macro_counter = 0
    current_macro: MacroDecision | None = None
    reject_count = 0
    timeout_count = 0
    decision_chunks: dict[str, dict[str, object]] = {}

    while not done and state.step < state.max_steps:
        obs_prev = build_task1_observation(state)
        if current_macro is None or current_macro.ttl_steps <= 0:
            l2_action = l2_policy.decide_action(obs_prev)
            seed_delta_q = adapt_action_delta_q(l2_action, n_joints=cfg.n_joints, max_delta_q=cfg.max_delta_q)
            decision_id = f"ep{episode_index}_d{macro_counter}"
            macro_counter += 1
            state_version += 1
            target_q = np.clip(state.q + seed_delta_q * float(decision_ttl_steps), -np.inf, np.inf)
            current_macro = MacroDecision(
                decision_id=decision_id,
                state_version=state_version,
                ttl_steps=decision_ttl_steps,
                target_q=target_q,
                seed_delta_q=seed_delta_q,
            )
            if hasattr(l3_executor, "execute_macro_kickoff") and bool((episode_debug_meta or {}).get("macro_kickoff_action", False)):
                l3_executor.execute_macro_kickoff(state, seed_delta_q)

        micro_delta = np.clip(current_macro.target_q - state.q, -cfg.max_delta_q, cfg.max_delta_q)
        l3_result = l3_executor.execute_with_safety(state, micro_delta)

        state = Task1State(
            q=l3_result.q_next.copy(),
            dq=l3_result.dq_next.copy(),
            target_pose_xyz=state.target_pose_xyz,
            step=state.step + 1,
            max_steps=state.max_steps,
            safe_z_min=state.safe_z_min,
            ee_proxy_xyz=l3_result.ee_proxy_xyz.copy() if l3_result.ee_proxy_xyz is not None else None,
        )
        obs_next = build_task1_observation(state)
        current_macro = MacroDecision(
            decision_id=current_macro.decision_id,
            state_version=current_macro.state_version,
            ttl_steps=current_macro.ttl_steps - 1,
            target_q=current_macro.target_q,
            seed_delta_q=current_macro.seed_delta_q,
        )

        done, success, term_reason = check_done_success(state, obs_next, safety_violation=l3_result.safety_violation, cfg=cfg)
        step_reason = "accepted" if l3_result.accepted else "rejected"
        if l3_result.safety_violation > 0.0:
            step_reason = f"{step_reason}:unsafe"
        if not l3_result.accepted:
            reject_count += 1
        if any("timeout" in log for log in l3_result.logs):
            timeout_count += 1

        reward = compose_task1_reward(
            mode=reward_mode,
            obs_prev=obs_prev,
            obs_next=obs_next,
            delta_q_cmd=micro_delta,
            safety_violation=l3_result.safety_violation,
            done=done,
            success=success,
            cfg=cfg,
        )
        total_reward += reward
        replay.append(ReplayTransition(d_pos_prev=obs_prev.d_pos, d_pos_next=obs_next.d_pos, reward=reward))

        chunk = decision_chunks.setdefault(
            current_macro.decision_id,
            {
                "decision_id": current_macro.decision_id,
                "state_version": current_macro.state_version,
                "ttl_steps": decision_ttl_steps,
                "steps": 0,
                "reward_sum": 0.0,
                "d_pos_prev": float(obs_prev.d_pos),
                "d_pos_next": float(obs_next.d_pos),
            },
        )
        chunk["steps"] = int(chunk["steps"]) + 1
        chunk["reward_sum"] = float(chunk["reward_sum"]) + float(reward)
        chunk["d_pos_next"] = float(obs_next.d_pos)

        traj.append(
            {
                "step": state.step,
                "obs": obs_next.to_dict(),
                "delta_q_cmd": micro_delta.tolist(),
                "reward": reward,
                "done": done,
                "success": success,
                "term_reason": term_reason,
                "decision_id": current_macro.decision_id,
                "state_version": current_macro.state_version,
                "decision_ttl": current_macro.ttl_steps,
                "l3_logs": list(l3_result.logs),
                "accepted": bool(l3_result.accepted),
                "debug": {
                    "ee_pos": state.ee_pos.tolist(),
                    "z_margin": float(obs_next.z_margin),
                    "d_pos": float(obs_next.d_pos),
                    "reason": step_reason,
                },
            }
        )

    decision_replay = [
        {
            **chunk,
            "reward_avg": float(chunk["reward_sum"]) / max(1, int(chunk["steps"])),
        }
        for chunk in decision_chunks.values()
    ]
    micro_steps = len(traj)
    reject_rate = float(reject_count / micro_steps) if micro_steps else 0.0
    episode_summary = {
        "macro_decisions": len(decision_replay),
        "micro_steps": micro_steps,
        "reject_count": reject_count,
        "timeout_count": timeout_count,
        "reject_rate": reject_rate,
    }

    return {
        "episode_index": episode_index,
        "reward_mode": reward_mode,
        "safe_z_min": float(cfg.safe_z_min),
        "episode_debug": episode_debug_meta or {},
        "total_reward": float(total_reward),
        "steps": micro_steps,
        "success": bool(success),
        "term_reason": term_reason,
        "target_pose_xyz": state.target_pose_xyz.tolist(),
        "trajectory": traj,
        "replay": [asdict(r) for r in replay],
        "replay_by_decision": decision_replay,
        "episode_summary": episode_summary,
    }


def _save_checkpoint(path: Path, *, l2: LearnableL2Policy, episodes: list[dict[str, object]], backend: RuntimeBackend) -> Path:
    payload = {
        "backend": backend,
        "l2": l2.to_checkpoint(),
        "episodes": [{"episode_index": ep["episode_index"], "total_reward": ep["total_reward"], "success": ep["success"]} for ep in episodes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def run_task1_training(
    *,
    episodes: int,
    reward_mode: RewardMode,
    backend: RuntimeBackend = "bootstrap",
    cfg: Task1Config | None = None,
    checkpoint_path: str | None = None,
    auto_reset: bool | None = None,
    reset_timeout: float = 4.0,
    scene_reset_cmd: str | None = None,
    allow_ee_fallback: bool = False,
    verbose_debug: bool = False,
    l3_exec_mode: Literal["action", "topic", "hybrid"] = "action",
    action_timeout: float = 1.5,
    action_server_timeout: float = 3.0,
    macro_ttl_steps: int = 4,
    macro_kickoff_action: bool = False,
    target_pose_xyz: np.ndarray | None = None,
) -> list[dict[str, object]]:
    config = cfg or Task1Config()
    l1 = HighPoseTargetProvider(target_xyz=np.asarray(target_pose_xyz, dtype=float)) if target_pose_xyz is not None else HighPoseTargetProvider()
    l2 = LearnableL2Policy()

    runtime_l3: GazeboRuntimeL3Executor | None = None
    if backend == "bootstrap":
        l3: L3ExecutorContract = SafetyConstrainedL3Executor(max_dq_per_step=config.max_delta_q)
    elif backend == "gazebo":
        runtime_l3 = GazeboRuntimeL3Executor(
            max_dq_per_step=config.max_delta_q,
            reset_timeout_sec=reset_timeout,
            scene_reset_cmd=scene_reset_cmd,
            allow_ee_fallback=allow_ee_fallback,
            verbose_debug=verbose_debug,
            l3_exec_mode=l3_exec_mode,
            action_timeout_sec=float(action_timeout),
            action_server_timeout_sec=float(action_server_timeout),
            macro_kickoff_action=bool(macro_kickoff_action),
        )
        l3 = runtime_l3
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    rows: list[dict[str, object]] = []
    replay_bank: list[ReplayTransition] = []
    gazebo_auto_reset = bool(auto_reset) if auto_reset is not None else backend == "gazebo"
    runtime_joint_names = (
        runtime_l3.runtime_joint_names(config.n_joints)
        if runtime_l3 is not None
        else tuple(f"joint_{idx}" for idx in range(config.n_joints))
    )
    if verbose_debug:
        print(
            f"[startup] backend={backend} resolved_n_joints={config.n_joints} "
            f"runtime_joint_names={list(runtime_joint_names)}"
        )
    try:
        for i in range(max(1, int(episodes))):
            initial_q = None
            initial_dq = None
            initial_ee_proxy = None
            reset_meta: dict[str, object] | None = None
            if runtime_l3 is not None:
                if i == 0:
                    initial_q, initial_dq, initial_ee_proxy = runtime_l3.read_runtime_state(n_joints=config.n_joints)
                    reset_meta = {
                        "applied": False,
                        "success": True,
                        "duration_ms": 0.0,
                        "initial_state": {
                            "q": initial_q.tolist(),
                            "dq": initial_dq.tolist(),
                            "ee_pose": initial_ee_proxy.tolist() if initial_ee_proxy is not None else None,
                            "ee_source": runtime_l3.ee_source_name(),
                        },
                        "error": None,
                        "skipped_reason": "episode_0_bootstrap_runtime_state",
                    }
                elif gazebo_auto_reset:
                    try:
                        reset_meta = runtime_l3.reset_episode(n_joints=config.n_joints, timeout_sec=reset_timeout)
                        initial_state = reset_meta.get("initial_state")
                        if not isinstance(initial_state, dict):
                            raise RuntimeError("backend=gazebo fail-fast: reset_episode returned invalid initial_state")
                        initial_q = np.asarray(initial_state.get("q"), dtype=float)
                        initial_dq = np.asarray(initial_state.get("dq"), dtype=float)
                        ee_proxy_raw = initial_state.get("ee_pose", initial_state.get("ee_proxy"))
                        initial_ee_proxy = None if ee_proxy_raw is None else np.asarray(ee_proxy_raw, dtype=float)
                    except Exception as exc:
                        reset_meta = getattr(exc, "reset_meta", None)
                        if not isinstance(reset_meta, dict):
                            reset_meta = {
                                "applied": True,
                                "success": False,
                                "duration_ms": 0.0,
                                "initial_state": None,
                                "error": str(exc),
                            }
                        rows.append({"episode_index": i, "backend": backend, "reset": reset_meta})
                        raise RuntimeError(
                            "backend=gazebo fail-fast: episode reset failed "
                            f"before episode {i}: {reset_meta.get('error') or exc}"
                        ) from exc
                else:
                    initial_q, initial_dq, initial_ee_proxy = runtime_l3.read_runtime_state(n_joints=config.n_joints)
                    reset_meta = {
                        "applied": False,
                        "success": True,
                        "duration_ms": 0.0,
                        "initial_state": {
                            "q": initial_q.tolist(),
                            "dq": initial_dq.tolist(),
                            "ee_pose": initial_ee_proxy.tolist() if initial_ee_proxy is not None else None,
                            "ee_source": runtime_l3.ee_source_name(),
                        },
                        "error": None,
                        "skipped_reason": "auto_reset_disabled",
                    }

            row = run_task1_episode(
                episode_index=i,
                reward_mode=reward_mode,
                cfg=config,
                l1_provider=l1,
                l2_policy=l2,
                l3_executor=l3,
                initial_q=initial_q,
                initial_dq=initial_dq,
                initial_ee_proxy_xyz=initial_ee_proxy,
                verbose_debug=verbose_debug,
                episode_debug_meta={
                    "backend": backend,
                    "safe_z_min": float(config.safe_z_min),
                    "ee_source": runtime_l3.ee_source_name() if runtime_l3 is not None else "bootstrap_q_proxy",
                    "reset_summary": reset_meta,
                    "macro_ttl_steps": int(macro_ttl_steps),
                    "macro_kickoff_action": bool(macro_kickoff_action),
                },
            )
            row["backend"] = backend
            if reset_meta is not None:
                row["reset"] = reset_meta
            rows.append(row)

            decision_replay_samples = [
                ReplayTransition(
                    d_pos_prev=float(sample["d_pos_prev"]),
                    d_pos_next=float(sample["d_pos_next"]),
                    reward=float(sample["reward_avg"]),
                )
                for sample in row.get("replay_by_decision", [])
            ]
            replay_samples = decision_replay_samples or [ReplayTransition(**sample) for sample in row.get("replay", [])]
            replay_bank.extend(replay_samples)
            l2.update_from_replay(replay_bank[-64:])
            row["l2_gain_after_update"] = l2.gain

        if checkpoint_path:
            ckpt = _save_checkpoint(Path(checkpoint_path), l2=l2, episodes=rows, backend=backend)
            for row in rows:
                row["checkpoint_path"] = str(ckpt)

        return rows
    finally:
        if runtime_l3 is not None:
            runtime_l3.close()


def _resolve_n_joints_for_backend(*, backend: RuntimeBackend, cli_n_joints: int | None) -> int:
    if cli_n_joints is not None:
        return int(cli_n_joints)
    if backend == "gazebo":
        return 7
    return int(Task1Config.n_joints)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V5 Task-1 EE pose reaching training loop")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--reward-mode", choices=["no_shaping", "heuristic", "pbrs"], default="heuristic")
    parser.add_argument("--backend", choices=["bootstrap", "gazebo"], default="bootstrap")
    parser.add_argument("--auto-reset", dest="auto_reset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reset-timeout", type=float, default=4.0)
    parser.add_argument("--scene-reset-cmd", type=str, default=None)
    parser.add_argument("--artifact-output", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--safe-z-min", type=float, default=Task1Config.safe_z_min)
    parser.add_argument("--n-joints", type=int, default=None, help="Action/state joint dimension used by the trainer")
    parser.add_argument("--allow-ee-fallback", action="store_true", help="Allow q[:3] fallback when no EE pose topic/TF is available")
    parser.add_argument("--l3-exec-mode", choices=["action", "topic", "hybrid"], default="action")
    parser.add_argument("--action-timeout", type=float, default=1.5)
    parser.add_argument("--action-server-timeout", type=float, default=3.0)
    parser.add_argument("--macro-ttl-steps", type=int, default=4)
    parser.add_argument("--macro-kickoff-action", action="store_true")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-z", type=float, default=None)
    parser.add_argument("--verbose-debug", action="store_true")
    args = parser.parse_args(argv)

    resolved_n_joints = _resolve_n_joints_for_backend(backend=args.backend, cli_n_joints=args.n_joints)
    cfg = Task1Config(n_joints=resolved_n_joints, safe_z_min=float(args.safe_z_min))

    target_pose_xyz = None
    if args.target_x is not None and args.target_y is not None and args.target_z is not None:
        target_pose_xyz = np.array([float(args.target_x), float(args.target_y), float(args.target_z)], dtype=float)

    rows = run_task1_training(
        episodes=args.episodes,
        reward_mode=args.reward_mode,
        backend=args.backend,
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        auto_reset=args.auto_reset,
        reset_timeout=args.reset_timeout,
        scene_reset_cmd=args.scene_reset_cmd,
        allow_ee_fallback=bool(args.allow_ee_fallback),
        verbose_debug=bool(args.verbose_debug),
        l3_exec_mode=args.l3_exec_mode,
        action_timeout=float(args.action_timeout),
        action_server_timeout=float(args.action_server_timeout),
        macro_ttl_steps=int(args.macro_ttl_steps),
        macro_kickoff_action=bool(args.macro_kickoff_action),
        target_pose_xyz=target_pose_xyz,
    )
    success_count = sum(1 for row in rows if row["success"])
    mean_reward = float(np.mean([float(row["total_reward"]) for row in rows]))
    print(
        f"backend={args.backend} episodes={len(rows)} "
        f"reward_mode={args.reward_mode} safe_z_min={args.safe_z_min:.3f} "
        f"success={success_count}/{len(rows)} mean_reward={mean_reward:.4f}"
    )

    if args.artifact_output:
        output_path = Path(args.artifact_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"artifact_output={output_path}")
    if args.checkpoint_path:
        print(f"checkpoint_output={args.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
