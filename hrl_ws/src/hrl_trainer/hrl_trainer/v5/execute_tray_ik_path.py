"""Execute the external tray holder-to-holder IK path through Phase 3A L3.

The external kitchen scene already contains a hand-authored tray waypoint
library and IK solver.  The original node publishes one large
``JointTrajectory`` topic message, but the current Phase 3A stack has been
validated through the action-primary L3 path.  This runner bridges the two:

``MoveTaskLibrary`` poses -> external IK q path -> RuntimeROS2Adapter action
goals -> Gazebo.

It is intended for headless demo validation and writes per-waypoint logs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from hrl_trainer.v5_1.runtime_ros2 import RuntimeROS2Adapter


REPO_ROOT = Path(__file__).resolve().parents[5]
EXTERNAL_CONTROLLER_SRC = REPO_ROOT / "external" / "kitchen_scene" / "src" / "kitchen_robot_controller"
if str(EXTERNAL_CONTROLLER_SRC) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_CONTROLLER_SRC))

from kitchen_robot_controller.kinematics import fk_ur, solve_ik  # noqa: E402
from kitchen_robot_controller.task_library import MoveTaskLibrary  # noqa: E402


JOINT_NAMES = [
    "Rack_joint",
    "robot_base_joint",
    "shoulder1_joint",
    "shoulder2_joint",
    "wr1_joint",
    "wr2_joint",
    "wr3_joint",
]


def _wrap_revolute_subset_in_rad(q: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(q, dtype=float).copy()
    for idx in (1, 2, 3, 4, 6):
        wrapped[idx] = (wrapped[idx] + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped


def postprocess_trajectory(q_list: Sequence[Sequence[float]]) -> list[np.ndarray]:
    if not q_list:
        return []
    two_pi = 2.0 * math.pi
    wrap_indices = [1, 2, 3, 4, 5, 6]
    processed: list[np.ndarray] = []
    prev: list[float] | None = None
    for raw_q in q_list:
        q = [float(v) for v in raw_q]
        if prev is None:
            for j in wrap_indices:
                q[j] = (q[j] + math.pi) % two_pi - math.pi
            processed.append(np.asarray(q, dtype=float))
            prev = q
            continue
        q_adj = q.copy()
        for j in wrap_indices:
            val = q_adj[j]
            delta = val - prev[j]
            while delta > math.pi:
                val -= two_pi
                delta = val - prev[j]
            while delta < -math.pi:
                val += two_pi
                delta = val - prev[j]
            q_adj[j] = val
        processed.append(np.asarray(q_adj, dtype=float))
        prev = q_adj
    return processed


def matrix_to_rpy(T: np.ndarray) -> tuple[float, float, float]:
    R = T[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        return (
            math.atan2(R[2, 1], R[2, 2]),
            math.atan2(-R[2, 0], sy),
            math.atan2(R[1, 0], R[0, 0]),
        )
    return (math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0)


def rotation_error_rad(target_R: np.ndarray, actual_R: np.ndarray) -> float:
    R_err = target_R @ actual_R.T
    val = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(val))


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    ori = rotation_error_rad(target[:3, :3], actual[:3, :3])
    return pos, ori


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def kinematic_tray_pose(
    *,
    progress: float,
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    z_base: float,
    lift_height: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Return an oracle tray pose for visual carry demos.

    The scene currently has no gripper/attachment model.  This pose is therefore
    an explicit kinematic attachment visualization: the arm follows the IK path,
    while the tray entity is moved along a smooth source-to-target arc.
    """

    t = smoothstep(progress)
    x = (1.0 - t) * float(source_xy[0]) + t * float(target_xy[0])
    y = (1.0 - t) * float(source_xy[1]) + t * float(target_xy[1])
    z = float(z_base) + max(0.0, float(lift_height)) * math.sin(math.pi * t)
    return (x, y, z, 0.0, 0.0, 0.0, 1.0)


def set_gz_entity_pose(
    *,
    entity_name: str,
    pose_xyzw: tuple[float, float, float, float, float, float, float],
    world_name: str,
    timeout_ms: int = 1000,
) -> dict[str, Any]:
    x, y, z, qx, qy, qz, qw = pose_xyzw
    req = (
        f'name: "{entity_name}" '
        f'position {{ x: {x:.6f} y: {y:.6f} z: {z:.6f} }} '
        f'orientation {{ x: {qx:.8f} y: {qy:.8f} z: {qz:.8f} w: {qw:.8f} }}'
    )
    cmd = [
        "gz",
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        "gz.msgs.Pose",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        str(int(timeout_ms)),
        "--req",
        req,
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=max(1.0, timeout_ms / 1000.0 + 0.5))
    stdout = completed.stdout or ""
    return {
        "entity_name": entity_name,
        "success": completed.returncode == 0 and "data: false" not in stdout.lower(),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": completed.stderr or "",
        "pose_xyzw": list(pose_xyzw),
    }


def build_ik_path(
    *,
    prop: str,
    src_index: int,
    dst_index: int,
    n_interp: int,
    initial_q: Sequence[float],
    include_return_home: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    task_lib = MoveTaskLibrary(n_interp=max(1, int(n_interp)))
    poses = task_lib.move_from_to(prop=prop, src_idx=int(src_index), dst_idx=int(dst_index))
    q_current = np.asarray(initial_q, dtype=float)
    raw_qs: list[np.ndarray] = []
    kept_poses: list[np.ndarray] = []
    for pose in poses:
        ik = solve_ik(pose, q_current)
        if not getattr(ik, "converged", True):
            raise RuntimeError(f"IK failed after {getattr(ik, 'iterations', 'unknown')} iterations")
        q_current = np.asarray(ik.q, dtype=float)
        raw_qs.append(q_current.copy())
        kept_poses.append(np.asarray(pose, dtype=float))

    qs = postprocess_trajectory(raw_qs)
    if include_return_home:
        qs.append(np.zeros(7, dtype=float))
        kept_poses.append(fk_ur(np.zeros(7, dtype=float)))
    return kept_poses, qs


def downsample_indices(length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    stride = max(1, int(stride))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def execute_path(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = Path(args.artifact_root)
    run_dir = artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    poses, qs = build_ik_path(
        prop=args.prop,
        src_index=args.src_index,
        dst_index=args.dst_index,
        n_interp=args.intermediate_points_per_segment,
        initial_q=[float(v) for v in args.initial_joint_positions.split(",")],
        include_return_home=not args.no_return_home,
    )
    indices = downsample_indices(len(qs), args.execute_stride)
    selected = [(i, poses[i], qs[i]) for i in indices]

    runtime = RuntimeROS2Adapter.from_ros2(
        joint_names=JOINT_NAMES,
        trajectory_topic=args.trajectory_topic,
        joint_state_topic=args.joint_state_topic,
        command_duration_s=args.command_duration_s,
        settle_timeout_s=args.settle_timeout_s,
        initial_warmup_timeout_s=args.initial_warmup_timeout_s,
        use_action_primary=not args.topic_only,
        gz_visualize_target=True,
        gz_world_name=args.gz_world_name,
        gz_target_entity_name=args.gz_target_entity_name,
        gz_target_world_offset_xyz=(0.0, 0.0, 0.0),
    )
    step_log = run_dir / "tray_ik_execution_steps.jsonl"
    if step_log.exists():
        step_log.unlink()
    rows: list[dict[str, Any]] = []
    try:
        for seq_idx, (source_idx, target_pose, q_target) in enumerate(selected):
            progress = float(seq_idx) / float(max(1, len(selected) - 1))
            tray_visual_result: dict[str, Any] | None = None
            if args.kinematic_carry_tray:
                tray_pose = kinematic_tray_pose(
                    progress=progress,
                    source_xy=(args.tray_source_x, args.tray_source_y),
                    target_xy=(args.tray_target_x, args.tray_target_y),
                    z_base=args.tray_z,
                    lift_height=args.tray_lift_height,
                )
                tray_visual_result = set_gz_entity_pose(
                    entity_name=args.tray_entity_name,
                    pose_xyzw=tray_pose,
                    world_name=args.gz_world_name,
                    timeout_ms=args.gz_service_timeout_ms,
                )
            target_pose6 = np.asarray([*target_pose[:3, 3].tolist(), *matrix_to_rpy(target_pose)], dtype=float)
            visual = runtime.publish_ee_target_visual(target_pose6)
            q_before = runtime.read_q(timeout_s=args.settle_timeout_s)
            before_pose = fk_ur(q_before)
            before_pos_err, before_ori_err = pose_error(target_pose, before_pose)
            result = runtime.step(np.asarray(q_target, dtype=float))
            q_after = np.asarray(result["q_after"], dtype=float)
            after_pose = fk_ur(q_after)
            after_pos_err, after_ori_err = pose_error(target_pose, after_pose)
            row = {
                "schema_version": "v5.phase3a.tray_ik_execution.step.v1",
                "sequence_index": seq_idx,
                "source_path_index": source_idx,
                "target_pose_xyz": target_pose[:3, 3].tolist(),
                "target_pose_rpy": list(matrix_to_rpy(target_pose)),
                "q_target": q_target.tolist(),
                "q_before": q_before.tolist(),
                "q_after": q_after.tolist(),
                "position_error_before": before_pos_err,
                "orientation_error_before": before_ori_err,
                "position_error_after": after_pos_err,
                "orientation_error_after": after_ori_err,
                "execution": result,
                "visualization": visual,
                "kinematic_carry_tray": bool(args.kinematic_carry_tray),
                "tray_visualization": tray_visual_result,
            }
            rows.append(row)
            with step_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        runtime.close()

    execution_ok_count = sum(1 for row in rows if row["execution"].get("execution_ok"))
    final_row = rows[-1] if rows else None
    summary = {
        "schema_version": "v5.phase3a.tray_ik_execution.summary.v1",
        "run_id": args.run_id,
        "prop": args.prop,
        "src_index": args.src_index,
        "dst_index": args.dst_index,
        "raw_path_points": len(qs),
        "executed_points": len(rows),
        "execute_stride": args.execute_stride,
        "execution_ok_rate": execution_ok_count / max(1, len(rows)),
        "max_position_error_after": max((row["position_error_after"] for row in rows), default=None),
        "max_orientation_error_after": max((row["orientation_error_after"] for row in rows), default=None),
        "mean_position_error_after": float(np.mean([row["position_error_after"] for row in rows])) if rows else None,
        "mean_orientation_error_after": float(np.mean([row["orientation_error_after"] for row in rows])) if rows else None,
        "final_position_error_after": final_row["position_error_after"] if final_row else None,
        "final_orientation_error_after": final_row["orientation_error_after"] if final_row else None,
        "step_log_path": str(step_log),
        "kinematic_carry_tray": bool(args.kinematic_carry_tray),
        "tray_entity_name": args.tray_entity_name if args.kinematic_carry_tray else None,
    }
    (run_dir / "tray_ik_execution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts/v5/phase3a_tray_ik_execution"))
    parser.add_argument("--run-id", default="tray_ik_holder1_to_8_001")
    parser.add_argument("--prop", default="tray")
    parser.add_argument("--src-index", type=int, default=1)
    parser.add_argument("--dst-index", type=int, default=8)
    parser.add_argument("--intermediate-points-per-segment", type=int, default=3)
    parser.add_argument("--execute-stride", type=int, default=4)
    parser.add_argument("--initial-joint-positions", default="0,0,0,0,0,0,0")
    parser.add_argument("--no-return-home", action="store_true")
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--command-duration-s", type=float, default=0.35)
    parser.add_argument("--settle-timeout-s", type=float, default=1.4)
    parser.add_argument("--initial-warmup-timeout-s", type=float, default=3.0)
    parser.add_argument("--topic-only", action="store_true")
    parser.add_argument("--gz-world-name", default="empty")
    parser.add_argument("--gz-target-entity-name", default="phase3a_tray_ik_target_marker")
    parser.add_argument("--gz-service-timeout-ms", type=int, default=1000)
    parser.add_argument("--kinematic-carry-tray", action="store_true")
    parser.add_argument("--tray-entity-name", default="tray1")
    parser.add_argument("--tray-source-x", type=float, default=-0.931843)
    parser.add_argument("--tray-source-y", type=float, default=-1.145751)
    parser.add_argument("--tray-target-x", type=float, default=-0.931843)
    parser.add_argument("--tray-target-y", type=float, default=1.176249)
    parser.add_argument("--tray-z", type=float, default=1.208908)
    parser.add_argument("--tray-lift-height", type=float, default=0.12)
    args = parser.parse_args()
    print(json.dumps(execute_path(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
