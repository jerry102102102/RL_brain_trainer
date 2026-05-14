"""Live Phase 3A demo orchestrator.

This module is built for screen recording, not offline reporting. It prints a
clear L1 -> L2 -> L3 story, resolves a natural-language command through the
Qwen/L1 bridge or a saved fallback artifact, publishes demo status, launches
target markers, and optionally runs the Gazebo RL controlled-sim path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from hrl_trainer.kinematic_phase1.kinematics.fk_interface import compute_ee_pose6

from .qwen_l1_client import run_l1_to_rl_input
from .tray_waypoint_plan import build_control_targets, build_vlm_plan, default_tray_carry_waypoints


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COMMAND = "Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose."
DEFAULT_FALLBACK_JSON = REPO_ROOT / "artifacts" / "v5" / "qwen_l1_demo" / "l1_to_rl_skill_request_qwen.json"
DEFAULT_HOLDER_ROUTE_TARGETS = (
    REPO_ROOT
    / "artifacts"
    / "kinematic_phase1"
    / "phase1c"
    / "scene_route_curriculum"
    / "tray1_holder1_to_8_route_q_goal_probe20.json"
)
DEFAULT_HOLDER_ROUTE_REGISTRY = (
    REPO_ROOT
    / "artifacts"
    / "kinematic_phase1"
    / "phase1c"
    / "approach_scene_route_curriculum_tray1_h1_to_h8_3m_001"
    / "phase3a"
    / "phase3a_runtime_models_routecurr.yaml"
)
DEFAULT_TRAY_LIKE_FALLBACK_SUMMARY = (
    REPO_ROOT
    / "artifacts"
    / "kinematic_phase1"
    / "workspace_expansion"
    / "tray_like_local_semantic_route_check_20260506"
    / "tray_like_local_semantic_route_summary.json"
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _print_section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _print_chain(stage: str, detail: str) -> None:
    print(f"[CHAIN] {stage:<8} | {detail}", flush=True)


def _log_status(status: str, log_path: Path) -> None:
    line = f"{time.strftime('%H:%M:%S')} | {status}"
    print(f"[STATUS] {status}", flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _ros_augmented_env() -> dict[str, str]:
    """Let the repo venv see ROS2 Python packages such as rclpy."""
    env = dict(os.environ)
    ros_py = "/opt/ros/jazzy/lib/python3.12/site-packages"
    current = env.get("PYTHONPATH", "")
    parts = [p for p in current.split(":") if p]
    if ros_py not in parts:
        parts.insert(0, ros_py)
    env["PYTHONPATH"] = ":".join(parts)
    ros_lib = "/opt/ros/jazzy/lib"
    ld_parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    if ros_lib not in ld_parts:
        ld_parts.insert(0, ros_lib)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env.setdefault("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    return env


def _copy_or_write_l1(
    *,
    command: str,
    output_dir: Path,
    backend: str,
    use_qwen: bool,
    fallback_json: Path,
) -> dict[str, Any]:
    out = output_dir / "l1_bridge_result.json"
    if use_qwen:
        try:
            result = run_l1_to_rl_input(command, backend=backend, output_path=out)
            payload = result.to_dict()
            payload["l1_source"] = backend
            return payload
        except Exception as exc:
            print(f"Qwen runtime unavailable; using saved L1 bridge artifact fallback. reason={type(exc).__name__}: {exc}", flush=True)
    else:
        print("Qwen runtime disabled; using saved L1 bridge artifact fallback.", flush=True)

    if not fallback_json.exists():
        raise FileNotFoundError(f"L1 fallback artifact missing: {fallback_json}")
    payload = _load_json(fallback_json)
    payload["l1_source"] = "fallback_json"
    _write_json(out, payload)
    return payload


def _extract_intent_and_skill(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    intent_resolution = payload.get("intent_resolution")
    intent_packet = intent_resolution.get("intent_packet") if isinstance(intent_resolution, Mapping) else payload.get("intent_packet")
    skill_request = payload.get("skill_request", payload)
    if not isinstance(intent_packet, Mapping):
        raise ValueError("L1 payload missing intent_packet")
    if not isinstance(skill_request, Mapping):
        raise ValueError("L1 payload missing skill_request")
    return dict(intent_packet), dict(skill_request)


def _publish_status_once(status: str, topic: str = "/v5/demo/status") -> None:
    try:
        import rclpy
        from std_msgs.msg import String

        rclpy.init(args=None)
        node = rclpy.create_node("phase3a_demo_status_once")
        pub = node.create_publisher(String, topic, 10)
        msg = String()
        msg.data = status
        deadline = time.monotonic() + 0.35
        while time.monotonic() < deadline:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        # Terminal log is the primary recording-friendly status surface.
        return


def _start_marker_process(request_json: Path, output_dir: Path, duration_s: float, route_markers_json: Path | None = None) -> subprocess.Popen[str] | None:
    if shutil.which("ros2") is None:
        print("WARN: ros2 not found; target marker node not launched.", flush=True)
        return None
    cmd = [
        sys.executable,
        "-m",
        "hrl_trainer.v5.target_marker_node",
        "--request-json",
        str(request_json),
        "--duration",
        str(duration_s),
    ]
    if route_markers_json is not None:
        cmd += ["--route-markers-json", str(route_markers_json)]
    log = (output_dir / "target_marker_node.log").open("w", encoding="utf-8")
    print("Starting target marker publisher:")
    print("  " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=_ros_augmented_env())


def _run_controlled_sim(
    output_dir: Path,
    *,
    run_id: str,
    max_targets: int,
    target_profile: str,
    approach_steps: int,
    finisher_steps: int,
    registry: Path | None = None,
    targets_json: Path | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "hrl_trainer.v5.phase3a_controlled_sim",
        "--artifact-root",
        str(REPO_ROOT / "artifacts" / "v5" / "phase3a_controlled_sim"),
        "--run-id",
        run_id,
        "--max-targets",
        str(max_targets),
        "--policy-device",
        "cpu",
        "--command-duration-s",
        "0.52" if target_profile == "workspace_showcase" else "0.40",
        "--settle-timeout-s",
        "2.0" if target_profile == "workspace_showcase" else "1.6",
        "--approach-steps",
        str(approach_steps),
        "--finisher-steps",
        str(finisher_steps),
    ]
    if registry is not None:
        cmd += ["--registry", str(registry)]
    if targets_json is not None:
        cmd += ["--targets-json", str(targets_json)]
    else:
        cmd += ["--target-profile", target_profile]
    print("Launching RL controlled-sim runtime:")
    print("  " + " ".join(cmd), flush=True)
    stdout_log = output_dir / "controlled_sim_stdout.log"
    stderr_log = output_dir / "controlled_sim_stderr.log"
    stderr_log.write_text("", encoding="utf-8")
    with stdout_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=_ros_augmented_env(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print("[RL] " + line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"controlled sim failed with code {returncode}; see {output_dir}")
    summary_path = REPO_ROOT / "artifacts" / "v5" / "phase3a_controlled_sim" / run_id / "controlled_sim_summary.json"
    return _load_json(summary_path)


def _write_route_marker_points(route_targets_json: Path, output_dir: Path, *, max_targets: int) -> Path:
    """Create RViz route marker points from holder route q-goal targets."""
    payload = _load_json(route_targets_json)
    entries = payload.get("targets", payload)
    if not isinstance(entries, list):
        raise ValueError(f"Route target JSON must be a list or contain targets: {route_targets_json}")
    points: list[dict[str, Any]] = []
    for idx, raw in enumerate(entries[: max(1, int(max_targets))]):
        if not isinstance(raw, Mapping) or "q_goal" not in raw:
            continue
        pose6 = compute_ee_pose6(raw["q_goal"])
        points.append(
            {
                "name": str(raw.get("name", f"route_target_{idx}")),
                "xyz": [float(pose6[0]), float(pose6[1]), float(pose6[2])],
            }
        )
    out = output_dir / "holder1_to_8_route_markers.json"
    _write_json(out, {"source": str(route_targets_json), "points": points})
    return out


def _write_tray_like_targets_and_markers(output_dir: Path) -> tuple[Path, Path, Path]:
    """Build the mock-tray semantic waypoint plan and executable level pose targets."""
    waypoints = default_tray_carry_waypoints()
    plan = build_vlm_plan(
        instruction=DEFAULT_COMMAND,
        source_slot="shelf_A1",
        target_slot="shelf_B1",
        object_id="tray1",
        waypoints=waypoints,
    )
    targets = build_control_targets(waypoints)
    plan_path = output_dir / "vlm_tray_like_waypoint_plan.json"
    targets_path = output_dir / "tray_like_controlled_targets.json"
    markers_path = output_dir / "tray_like_route_markers.json"
    _write_json(plan_path, plan)
    _write_json(targets_path, targets)
    points: list[dict[str, Any]] = []
    for wp in waypoints:
        points.append(
            {
                "name": wp.name,
                "description": wp.description,
                "xyz": [float(wp.xyz[0]), float(wp.xyz[1]), float(wp.xyz[2])],
                "rpy": [float(wp.rpy[0]), float(wp.rpy[1]), float(wp.rpy[2])],
            }
        )
    _write_json(markers_path, {"source": str(targets_path), "points": points})
    return plan_path, targets_path, markers_path


def _sync_plan_descriptions_from_l1(plan_path: Path, skill_request: Mapping[str, Any]) -> None:
    semantic_subtasks = skill_request.get("semantic_subtasks", [])
    if not isinstance(semantic_subtasks, list) or not semantic_subtasks:
        return
    plan = _load_json(plan_path)
    waypoints = plan.get("waypoints")
    if not isinstance(waypoints, list):
        return
    by_name = {
        str(item.get("name")): item
        for item in semantic_subtasks
        if isinstance(item, Mapping) and item.get("name")
    }
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            continue
        item = by_name.get(str(waypoint.get("name")))
        if not isinstance(item, Mapping):
            continue
        if item.get("description"):
            waypoint["description"] = str(item["description"])
        if item.get("posture_constraint"):
            waypoint["posture_constraint"] = str(item["posture_constraint"])
    plan["l1_semantic_subtasks_used"] = True
    _write_json(plan_path, plan)


def _print_tray_like_plan_table(plan_path: Path, targets_path: Path, markers_path: Path) -> None:
    plan = _load_json(plan_path)
    targets = _load_json(targets_path)
    markers = _load_json(markers_path)
    waypoints = plan.get("waypoints", [])
    controlled_targets = targets.get("targets", [])
    marker_points = markers.get("points", [])

    print("\n[L1/MCP -> L2] Tray-like semantic subtasks and executable RL targets", flush=True)
    print("  L1/MCP provides semantic subtasks and constraints; it does not output raw actions or trajectories.", flush=True)
    print("  L2 maps those subtasks into level pose targets inside the learned RL workspace.", flush=True)
    print("  The shelf_B1 pose remains the L1 semantic goal; pose6 targets are only L2 RL observations.", flush=True)
    print("  Constraint: every waypoint keeps the EE tray plane horizontal to the table.", flush=True)
    print("  # | waypoint                | visual xyz (world)              | rpy target(rad)        | posture intent       | meaning", flush=True)
    print("  --+-------------------------+---------------------------------+------------------------+----------------------+-------------------------------", flush=True)
    for idx, waypoint in enumerate(waypoints, start=1):
        name = str(waypoint.get("name", f"waypoint_{idx}"))
        desc = str(waypoint.get("description", ""))
        target = controlled_targets[idx - 1] if idx - 1 < len(controlled_targets) else {}
        marker = marker_points[idx - 1] if idx - 1 < len(marker_points) else {}
        xyz = marker.get("xyz", [])
        rpy = marker.get("rpy", [])
        xyz_text = "[" + ", ".join(f"{float(v):+.3f}" for v in xyz[:3]) + "]" if isinstance(xyz, list) else "n/a"
        rpy_text = "[" + ", ".join(f"{float(v):+.2f}" for v in rpy[:3]) + "]" if isinstance(rpy, list) else "n/a"
        hold_level = bool(waypoint.get("hold_level", True))
        posture = "keep tray level" if hold_level else "pose transition"
        print(f"  {idx:>1} | {name:<23} | {xyz_text:<31} | {rpy_text:<22} | {posture:<20} | {desc}", flush=True)


def _print_tray_like_execution_summary(summary: Mapping[str, Any]) -> None:
    targets = summary.get("targets", [])
    print("\n[L2/L3] Tray-like execution result table", flush=True)
    print("  # | waypoint                | start pos/ori      | final pos/ori      | success | L3 path", flush=True)
    print("  --+-------------------------+--------------------+--------------------+---------+----------------", flush=True)
    if not isinstance(targets, list):
        return
    for idx, target in enumerate(targets, start=1):
        if not isinstance(target, Mapping):
            continue
        finisher = target.get("finisher", {})
        name = str(target.get("target_name", f"target_{idx}"))
        start_pos = float(target.get("start_position_error", float("nan")))
        start_ori = float(target.get("start_orientation_error", float("nan")))
        final_pos = float(finisher.get("final_position_error", float("nan"))) if isinstance(finisher, Mapping) else float("nan")
        final_ori = float(finisher.get("final_orientation_error", float("nan"))) if isinstance(finisher, Mapping) else float("nan")
        success = bool(target.get("final_success", False))
        # The detailed step log contains per-step command_path. The summary row is intentionally compact.
        print(
            f"  {idx:>1} | {name:<23} | {start_pos:>6.3f}m/{start_ori:>5.3f}r | "
            f"{final_pos:>6.3f}m/{final_ori:>5.3f}r | {str(success):<7} | /arm_controller",
            flush=True,
        )
    print(
        f"  mean final error: pos={float(summary.get('final_position_error_mean', float('nan'))):.4f}m, "
        f"ori={float(summary.get('final_orientation_error_mean', float('nan'))):.4f}rad, "
        f"success_rate={float(summary.get('success_rate', float('nan'))):.3f}",
        flush=True,
    )


def _write_tray_like_demo_report(output_dir: Path, plan_path: Path, summary: Mapping[str, Any]) -> Path:
    plan = _load_json(plan_path)
    lines = [
        "# Tray-Like Transport Demo Explanation",
        "",
        "This run is a mock tray transport demonstration inside the learned RL workspace.",
        "It is not claiming full holder1-to-holder8 transport. The point is to show the live L1 -> L2 -> L3 chain.",
        "",
        "## L1 Output",
        "",
        "- Input command: Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.",
        "- L1 resolves object/source/target/constraints and selects `APPROACH -> FINISHER`.",
        "- L1 does not output raw joint commands.",
        "",
        "## L2 Plan",
        "",
        "| # | Waypoint | Meaning |",
        "|---:|---|---|",
    ]
    for idx, waypoint in enumerate(plan.get("waypoints", []), start=1):
        lines.append(f"| {idx} | `{waypoint.get('name', f'waypoint_{idx}')}` | {waypoint.get('description', '')} |")

    lines += [
        "",
        "## L3/Gazebo Result",
        "",
        "| # | Waypoint | Start Pos / Ori | Final Pos / Ori | Success |",
        "|---:|---|---:|---:|---:|",
    ]
    for idx, target in enumerate(summary.get("targets", []) if isinstance(summary.get("targets"), list) else [], start=1):
        if not isinstance(target, Mapping):
            continue
        finisher = target.get("finisher", {})
        final_pos = float(finisher.get("final_position_error", float("nan"))) if isinstance(finisher, Mapping) else float("nan")
        final_ori = float(finisher.get("final_orientation_error", float("nan"))) if isinstance(finisher, Mapping) else float("nan")
        lines.append(
            f"| {idx} | `{target.get('target_name', f'target_{idx}')}` | "
            f"{float(target.get('start_position_error', float('nan'))):.4f} m / {float(target.get('start_orientation_error', float('nan'))):.4f} rad | "
            f"{final_pos:.4f} m / {final_ori:.4f} rad | {bool(target.get('final_success', False))} |"
        )
    lines += [
        "",
        f"- Success rate: {float(summary.get('success_rate', float('nan'))):.3f}",
        f"- Mean final position error: {float(summary.get('final_position_error_mean', float('nan'))):.4f} m",
        f"- Mean final orientation error: {float(summary.get('final_orientation_error_mean', float('nan'))):.4f} rad",
        "",
        "## Why This May Look Small In Gazebo",
        "",
        "The executable waypoints are deliberately local `q_delta` targets inside the learned controller workspace.",
        "The semantic L1 target is shelf_B1, but this demo does not claim full physical tray transport across the kitchen scene.",
    ]
    out = output_dir / "tray_like_demo_explanation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _run_tray_like_fallback(output_dir: Path) -> dict[str, Any]:
    summary = {
        "mode": "tray_like_fallback_kinematic_summary",
        "note": "GZ execution unavailable; showing the verified local tray-like semantic waypoint route summary.",
    }
    if DEFAULT_TRAY_LIKE_FALLBACK_SUMMARY.exists():
        summary.update(_load_json(DEFAULT_TRAY_LIKE_FALLBACK_SUMMARY))
    _write_json(output_dir / "tray_like_replay_summary.json", summary)
    return summary


def _run_route_prefix_fallback(output_dir: Path, prefix: int) -> dict[str, Any]:
    """Use the official route summaries as a stable route-prefix replay fallback."""
    prefix_path = REPO_ROOT / "artifacts" / "kinematic_phase1" / "route_curriculum" / "route_prefix120_routeobs_sequence2_1m_001" / "route_eval_sequential" / "route_eval_sequential_summary.json"
    full_path = REPO_ROOT / "artifacts" / "kinematic_phase1" / "route_curriculum" / "eval_prefix120_model_full483_001" / "route_eval_sequential_summary.json"
    summary = {
        "mode": "route_prefix_fallback_kinematic_replay",
        "requested_prefix": int(prefix),
        "note": "GZ route prefix execution is not yet stable; showing official kinematic route rollout evidence.",
    }
    if prefix_path.exists():
        p = _load_json(prefix_path)
        summary.update(
            {
                "prefix120_success_rate": p.get("success_rate"),
                "prefix120_longest_success_prefix": p.get("longest_success_prefix"),
                "prefix120_route_distance_m": p.get("cumulative_successful_route_distance_m"),
            }
        )
    if full_path.exists():
        f = _load_json(full_path)
        summary.update(
            {
                "full483_probe_success_rate": f.get("success_rate"),
                "full483_probe_longest_success_prefix": f.get("longest_success_prefix"),
                "first_failure_index": f.get("first_failure_index"),
                "first_failure_reason": f.get("first_failure_reason"),
            }
        )
    _write_json(output_dir / "route_prefix_replay_summary.json", summary)
    return summary


def run_live_demo(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root)
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"live_demo_{args.demo_mode}_{_timestamp()}"
    output_dir = out_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    status_log = output_dir / "runtime_status.log"
    command_path = output_dir / "command.txt"
    command_path.write_text(args.command + "\n", encoding="utf-8")

    _print_section("[1/6] Received User Command")
    print(args.command, flush=True)
    _print_chain("USER", "natural language task command received")
    _log_status("WAITING_FOR_COMMAND", status_log)
    _publish_status_once("WAITING_FOR_COMMAND")

    _print_section("[2/6] Calling Qwen / L1 Bridge")
    _print_chain("L1/MCP", "resolve_intent_packet -> prepare_phase1_skill_request")
    print("[L1/MCP] Safety boundary: L1 produces IntentPacket only; no raw joint commands.", flush=True)
    _log_status("L1_RESOLVING_INTENT", status_log)
    l1_payload = _copy_or_write_l1(
        command=args.command,
        output_dir=output_dir,
        backend=args.qwen_backend,
        use_qwen=args.use_qwen,
        fallback_json=Path(args.fallback_json),
    )
    intent_packet, skill_request = _extract_intent_and_skill(l1_payload)
    _write_json(output_dir / "intent_packet.json", intent_packet)
    _write_json(output_dir / "skill_request.json", skill_request)

    _print_section("[3/6] Resolved IntentPacket")
    print(f"object = {intent_packet.get('object_id')}")
    print(f"source = {intent_packet.get('source_slot')}")
    print(f"target = {intent_packet.get('target_slot')}")
    print(f"constraints = {intent_packet.get('constraints')}")
    print(f"selected pipeline = {skill_request.get('pipeline')}")
    print(f"target pose = {skill_request.get('target_pose')}")
    semantic_subtasks = skill_request.get("semantic_subtasks", [])
    if isinstance(semantic_subtasks, list) and semantic_subtasks:
        names = [str(item.get("name", "")) for item in semantic_subtasks if isinstance(item, Mapping)]
        print(f"[L1/MCP] semantic subtasks = {' -> '.join(name for name in names if name)}", flush=True)
    else:
        print("[L1/MCP] semantic subtasks = not provided by artifact; L2 will use demo subtask template", flush=True)
    print("[L1/MCP] Resolved command = MOVE_PLATE(shelf_A1, shelf_B1)", flush=True)
    _print_chain("L1->L2", "structured skill request handed to learned policy runtime")
    _log_status("INTENT_RESOLVED", status_log)
    _publish_status_once("INTENT_RESOLVED")

    route_markers_json: Path | None = None
    tray_like_plan_path: Path | None = None
    tray_like_targets_json: Path | None = None
    if args.demo_mode == "route_prefix":
        route_targets = Path(args.route_targets_json)
        if route_targets.exists():
            route_markers_json = _write_route_marker_points(route_targets, output_dir, max_targets=args.route_prefix)
        else:
            print(f"WARN: holder route targets missing; route markers disabled: {route_targets}", flush=True)
    elif args.demo_mode == "tray_like_transport":
        tray_like_plan_path, tray_like_targets_json, route_markers_json = _write_tray_like_targets_and_markers(output_dir)
        _sync_plan_descriptions_from_l1(tray_like_plan_path, skill_request)
        _print_tray_like_plan_table(tray_like_plan_path, tray_like_targets_json, route_markers_json)

    marker_proc: subprocess.Popen[str] | None = None
    try:
        _print_section("[4/6] Publishing Target Marker")
        _print_chain("VIS", "publishing target cube/sphere/axis/text marker on /v5/demo/target_marker")
        marker_proc = _start_marker_process(
            output_dir / "l1_bridge_result.json",
            output_dir,
            duration_s=args.marker_duration,
            route_markers_json=route_markers_json,
        )
        _log_status("TARGET_MARKER_PUBLISHED", status_log)
        _publish_status_once("TARGET_MARKER_PUBLISHED")

        final_summary: dict[str, Any]
        if args.demo_mode == "dry_run_l1":
            _print_section("[5/6] Dry Run Only")
            print("Dry-run mode: no Gazebo control requested.", flush=True)
            final_summary = {
                "demo_mode": args.demo_mode,
                "status": "DONE",
                "l1_source": l1_payload.get("l1_source", args.qwen_backend),
            }
        elif args.demo_mode == "local_skill":
            _print_section("[5/6] Starting RL Runtime")
            _print_chain("L2/RL", "frozen Approach policy starts from current /joint_states")
            print(
                f"[L2/RL] demo target profile = {args.target_profile}; "
                f"targets = {args.max_targets}; approach_steps = {args.approach_steps}; "
                f"finisher_steps = {args.finisher_steps}",
                flush=True,
            )
            if args.target_profile == "workspace_showcase":
                print(
                    "[L2/RL] Workspace showcase mode intentionally uses larger FK targets. "
                    "A target may fail; the recording goal is visible learned motion and the L1->L2->L3 chain.",
                    flush=True,
                )
            _log_status("APPROACH_RUNNING", status_log)
            _publish_status_once("APPROACH_RUNNING")
            summary = _run_controlled_sim(
                output_dir,
                run_id=f"{run_id}_controlled_sim",
                max_targets=args.max_targets,
                target_profile=args.target_profile,
                approach_steps=args.approach_steps,
                finisher_steps=args.finisher_steps,
            )
            _print_chain("L3/GZ", "joint trajectory commands executed through Gazebo controller")
            _log_status("FINISHER_RUNNING", status_log)
            _publish_status_once("FINISHER_RUNNING")
            final_summary = {
                "demo_mode": args.demo_mode,
                "status": "DONE",
                "controlled_sim_summary": summary,
            }
        elif args.demo_mode == "route_prefix":
            _print_section("[5/6] Holder1 -> Holder8 Route Prefix Demo")
            _log_status("ROUTE_PREFIX_RUNNING", status_log)
            _publish_status_once("ROUTE_PREFIX_RUNNING")
            route_targets = Path(args.route_targets_json)
            route_registry = Path(args.route_registry)
            print(f"[L2/RL] holder route targets = {route_targets}", flush=True)
            print(f"[L2/RL] route registry = {route_registry}", flush=True)
            print(f"[L2/RL] requested holder1->8 prefix targets = {args.route_prefix}", flush=True)
            try:
                route_summary = _run_controlled_sim(
                    output_dir,
                    run_id=f"{run_id}_holder1_to_8_route",
                    max_targets=args.route_prefix,
                    target_profile="smoke",
                    approach_steps=args.route_approach_steps,
                    finisher_steps=args.route_finisher_steps,
                    registry=route_registry,
                    targets_json=route_targets,
                )
                final_summary = {
                    "demo_mode": args.demo_mode,
                    "status": "DONE",
                    "route_execution": "gazebo_controlled_sim",
                    "route_summary": route_summary,
                }
            except Exception as exc:
                print(
                    "GZ holder route execution unavailable; showing official kinematic route rollout replay. "
                    f"reason={type(exc).__name__}: {exc}",
                    flush=True,
                )
                route_summary = _run_route_prefix_fallback(output_dir, prefix=args.route_prefix)
                final_summary = {
                    "demo_mode": args.demo_mode,
                    "status": "DONE_WITH_KINEMATIC_ROUTE_REPLAY",
                    "route_execution": "kinematic_replay_fallback",
                    "route_summary": route_summary,
                }
        elif args.demo_mode == "tray_like_transport":
            _print_section("[5/6] Mock Tray Transport Semantic Waypoint Demo")
            _log_status("TRAY_LIKE_TRANSPORT_RUNNING", status_log)
            _publish_status_once("TRAY_LIKE_TRANSPORT_RUNNING")
            print("[L1/MCP] Qwen/L1 semantic waypoint plan created:", flush=True)
            print(f"  plan = {tray_like_plan_path}", flush=True)
            print(f"  controlled targets = {tray_like_targets_json}", flush=True)
            print("[L2/RL] executing tray-like waypoints with Approach -> Finisher; q_delta is target observation, not direct control.", flush=True)
            try:
                tray_summary = _run_controlled_sim(
                    output_dir,
                    run_id=f"{run_id}_tray_like_transport",
                    max_targets=args.tray_like_waypoints,
                    target_profile="smoke",
                    approach_steps=args.tray_like_approach_steps,
                    finisher_steps=args.tray_like_finisher_steps,
                    targets_json=tray_like_targets_json,
                )
                _print_tray_like_execution_summary(tray_summary)
                report_path = _write_tray_like_demo_report(output_dir, tray_like_plan_path, tray_summary)
                print(f"[REPORT] tray-like readable report = {report_path}", flush=True)
                final_summary = {
                    "demo_mode": args.demo_mode,
                    "status": "DONE",
                    "route_execution": "gazebo_controlled_sim",
                    "readable_report_path": str(report_path),
                    "tray_like_plan_path": str(tray_like_plan_path),
                    "tray_like_summary": tray_summary,
                }
            except Exception as exc:
                print(
                    "GZ tray-like transport unavailable; showing verified kinematic waypoint summary. "
                    f"reason={type(exc).__name__}: {exc}",
                    flush=True,
                )
                tray_summary = _run_tray_like_fallback(output_dir)
                report_path = _write_tray_like_demo_report(output_dir, tray_like_plan_path, tray_summary)
                print(f"[REPORT] tray-like readable report = {report_path}", flush=True)
                final_summary = {
                    "demo_mode": args.demo_mode,
                    "status": "DONE_WITH_KINEMATIC_TRAY_LIKE_REPLAY",
                    "route_execution": "kinematic_replay_fallback",
                    "readable_report_path": str(report_path),
                    "tray_like_plan_path": str(tray_like_plan_path),
                    "tray_like_summary": tray_summary,
                }
        else:
            raise ValueError(f"Unsupported demo mode: {args.demo_mode}")

        _print_section("[6/6] Demo Finished")
        _log_status(str(final_summary["status"]), status_log)
        _publish_status_once(str(final_summary["status"]))
        final_summary.update(
            {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "command": args.command,
                "intent_packet_path": str(output_dir / "intent_packet.json"),
                "skill_request_path": str(output_dir / "skill_request.json"),
                "status_log": str(status_log),
            }
        )
        _write_json(output_dir / "final_summary.json", final_summary)
        print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
        return final_summary
    finally:
        if marker_proc is not None:
            marker_proc.terminate()
            try:
                marker_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                marker_proc.kill()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live VLM/L1 -> RL -> Gazebo demo")
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    parser.add_argument("--demo-mode", choices=["dry_run_l1", "local_skill", "route_prefix", "tray_like_transport"], default="local_skill")
    parser.add_argument("--use-qwen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qwen-backend", choices=["mock_qwen", "qwen_subprocess"], default="mock_qwen")
    parser.add_argument("--fallback-json", default=str(DEFAULT_FALLBACK_JSON))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "report" / "demo_outputs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--marker-duration", type=float, default=120.0)
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument(
        "--target-profile",
        choices=["smoke", "visible_workspace", "demo_showcase", "workspace_showcase", "recording_showcase"],
        default="smoke",
    )
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--finisher-steps", type=int, default=6)
    parser.add_argument("--route-prefix", type=int, default=40)
    parser.add_argument("--route-targets-json", default=str(DEFAULT_HOLDER_ROUTE_TARGETS))
    parser.add_argument("--route-registry", default=str(DEFAULT_HOLDER_ROUTE_REGISTRY))
    parser.add_argument("--route-approach-steps", type=int, default=10)
    parser.add_argument("--route-finisher-steps", type=int, default=8)
    parser.add_argument("--tray-like-waypoints", type=int, default=6)
    parser.add_argument("--tray-like-approach-steps", type=int, default=60)
    parser.add_argument("--tray-like-finisher-steps", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_live_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
