"""Controlled Phase 3A Gazebo validation for Approach -> Finisher policies.

This runner is intentionally small and conservative.  It reads the current
Gazebo/ROS2 joint state, creates reachable FK targets, visualizes each target in
Gazebo, rolls out frozen Approach and Finisher policies, sends commands through
the existing RuntimeROS2Adapter/L3 path, and writes step-level logs for
comparison against the kinematic training runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hrl_trainer.kinematic_phase1.envs.observation_builder import build_observation
from hrl_trainer.kinematic_phase1.kinematics.fk_interface import compute_ee_pose6
from hrl_trainer.kinematic_phase1.kinematics.joint_limits import (
    JOINT_ORDER,
    JointSpec,
    clip_joint_configuration,
    default_joint_specs,
    delta_limits,
)
from hrl_trainer.kinematic_phase1.kinematics.pose_utils import pose_error_components
from hrl_trainer.kinematic_phase1.training.policy_config import load_yaml_file, to_env_config
from hrl_trainer.v5.runtime_model_registry import Phase3ARuntimeRegistry, load_phase3a_model_registry
from hrl_trainer.v5_1.runtime_ros2 import RuntimeROS2Adapter


REPO_ROOT = Path(__file__).resolve().parents[5]

WAYPOINT_TASK_DESCRIPTIONS = {
    "pre_grasp_align": "align with the tray approach side while keeping the EE horizontal to the ground",
    "under_tray_insert_pose": "slide forward under the tray with the EE still horizontal",
    "level_lift": "lift while preserving a level tray-carry posture",
    "carry_midline": "carry across the local workspace while keeping the tray plane level",
    "pre_insert_align": "align with the destination insertion pose without tilting the EE",
    "stable_insert_hold": "hold the final insertion pose level so the finisher can settle",
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    goal_pose6: np.ndarray
    source: str
    q_goal: np.ndarray | None = None


@dataclass(frozen=True)
class PhaseRuntimeConfig:
    role: str
    checkpoint: Path
    action_delta_scale: float
    dynamic_action_delta_scale_enabled: bool
    dynamic_action_delta_scale_near_pos_threshold_m: float
    dynamic_action_delta_scale_far_pos_threshold_m: float
    dynamic_action_delta_scale_near_multiplier: float
    dynamic_action_delta_scale_far_multiplier: float
    episode_steps: int
    mode_index: int


def _norm(value: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float)))


def pose_error_norms(current_pose6: Sequence[float], goal_pose6: Sequence[float]) -> tuple[float, float]:
    pos_err, ori_err = pose_error_components(current_pose6, goal_pose6)
    return _norm(pos_err), _norm(ori_err)


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def task_description(target_name: str, target_index: int, target_count: int) -> str:
    description = WAYPOINT_TASK_DESCRIPTIONS.get(target_name)
    if description:
        return f"Task {target_index + 1}/{target_count}: {description}"
    return f"Task {target_index + 1}/{target_count}: move to waypoint {target_name}"


def load_sb3_policy(checkpoint: Path, *, device: str = "cpu") -> Any:
    errors: list[str] = []
    for algo_name in ("PPO", "TD3", "SAC"):
        try:
            module = __import__("stable_baselines3", fromlist=[algo_name])
            algo = getattr(module, algo_name)
            return algo.load(str(checkpoint), device=device)
        except Exception as exc:  # pragma: no cover - depends on installed algos/assets
            errors.append(f"{algo_name}: {exc}")
    raise RuntimeError(f"Could not load SB3 checkpoint {checkpoint}: {'; '.join(errors)}")


def phase_runtime_config(
    registry: Phase3ARuntimeRegistry,
    *,
    role: str,
    repo_root: Path = REPO_ROOT,
    override_steps: int | None = None,
) -> PhaseRuntimeConfig:
    asset = registry.approach if role == "APPROACH" else registry.finisher
    cfg = to_env_config(load_yaml_file(asset.resolve_config(repo_root)))
    mode_index = 0 if role == "APPROACH" else 1
    steps = int(override_steps) if override_steps is not None else int(cfg.episode_length)
    return PhaseRuntimeConfig(
        role=role,
        checkpoint=asset.resolve_checkpoint(repo_root),
        action_delta_scale=float(cfg.action_delta_scale),
        dynamic_action_delta_scale_enabled=bool(cfg.dynamic_action_delta_scale_enabled),
        dynamic_action_delta_scale_near_pos_threshold_m=float(cfg.dynamic_action_delta_scale_near_pos_threshold_m),
        dynamic_action_delta_scale_far_pos_threshold_m=float(cfg.dynamic_action_delta_scale_far_pos_threshold_m),
        dynamic_action_delta_scale_near_multiplier=float(cfg.dynamic_action_delta_scale_near_multiplier),
        dynamic_action_delta_scale_far_multiplier=float(cfg.dynamic_action_delta_scale_far_multiplier),
        episode_steps=steps,
        mode_index=mode_index,
    )


def action_to_command_q(
    *,
    q: np.ndarray,
    action: np.ndarray,
    joint_specs: Sequence[JointSpec],
    action_delta_scale: float,
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    max_delta_q = delta_limits(joint_specs) * float(action_delta_scale)
    return clip_joint_configuration(np.asarray(q, dtype=float) + action * max_delta_q, joint_specs)


def effective_action_delta_scale(phase_cfg: PhaseRuntimeConfig, pos_error: float) -> float:
    base_scale = float(phase_cfg.action_delta_scale)
    if not phase_cfg.dynamic_action_delta_scale_enabled:
        return base_scale
    near = float(phase_cfg.dynamic_action_delta_scale_near_pos_threshold_m)
    far = float(phase_cfg.dynamic_action_delta_scale_far_pos_threshold_m)
    if near <= 0.0 or far <= near:
        return base_scale
    near_mult = float(phase_cfg.dynamic_action_delta_scale_near_multiplier)
    far_mult = float(phase_cfg.dynamic_action_delta_scale_far_multiplier)
    if pos_error <= near:
        mult = near_mult
    elif pos_error >= far:
        mult = far_mult
    else:
        alpha = (pos_error - near) / max(far - near, 1e-9)
        mult = near_mult + alpha * (far_mult - near_mult)
    return float(base_scale * max(mult, 0.0))


def _predict_action(policy: Any, observation: Mapping[str, np.ndarray]) -> np.ndarray:
    action, _ = policy.predict(dict(observation), deterministic=True)
    return np.asarray(action, dtype=float).reshape(-1)


def build_default_targets(
    *,
    q_reference: np.ndarray,
    joint_specs: Sequence[JointSpec],
    max_targets: int,
    profile: str = "smoke",
) -> list[TargetSpec]:
    """Create reachable FK targets from deterministic joint-space offsets."""
    if profile == "smoke":
        deltas = [
            np.array([0.025, 0.040, -0.025, 0.030, 0.018, -0.018, 0.015], dtype=float),
            np.array([0.045, -0.045, 0.035, -0.030, -0.020, 0.020, -0.018], dtype=float),
            np.array([-0.035, 0.035, 0.025, 0.025, 0.016, -0.020, 0.020], dtype=float),
        ]
    elif profile in ("visible_workspace", "demo_showcase", "workspace_showcase", "recording_showcase"):
        # Larger but still conservative FK targets for visible Gazebo motion.
        # These stay well below joint limits while moving multiple arm axes.
        base = np.array(
            [
                [0.08, 0.18, -0.12, 0.13, 0.07, -0.08, 0.06],
                [0.10, -0.16, 0.13, -0.12, -0.08, 0.07, -0.07],
                [-0.08, 0.15, 0.12, 0.10, -0.06, -0.09, 0.08],
                [-0.10, -0.18, -0.10, -0.13, 0.09, 0.08, -0.06],
                [0.12, 0.10, -0.16, 0.08, 0.10, -0.06, 0.09],
                [-0.12, -0.11, 0.15, -0.08, -0.09, 0.06, -0.08],
                [0.06, 0.22, 0.06, -0.16, 0.08, 0.10, -0.10],
                [-0.06, -0.22, -0.06, 0.16, -0.08, -0.10, 0.10],
                [0.14, -0.06, -0.14, 0.14, -0.10, 0.08, 0.06],
                [-0.14, 0.06, 0.14, -0.14, 0.10, -0.08, -0.06],
            ],
            dtype=float,
        )
        deltas = []
        for cycle in range(2):
            scale = 1.0 if cycle == 0 else 1.25
            sign = 1.0 if cycle == 0 else -1.0
            deltas.extend([sign * scale * row for row in base])
        if profile == "demo_showcase":
            # Curated from the visible-workspace probe: these waypoints produce
            # clear motion while avoiding the one known weak target in the
            # first six visible-workspace samples.
            deltas = [deltas[i] for i in (0, 1, 2, 4, 5)]
        elif profile == "workspace_showcase":
            # Presentation-oriented targets: larger than the stable smoke/demo
            # profile so the movement is obvious in Gazebo. This mode is meant
            # for screen recording; a few targets may fail, and that is OK.
            stage8_like = np.array(
                [
                    [0.16, 0.30, -0.20, 0.22, 0.12, -0.12, 0.10],
                    [-0.16, -0.30, 0.20, -0.22, -0.12, 0.12, -0.10],
                    [0.20, -0.24, 0.22, -0.18, -0.13, 0.10, -0.11],
                    [-0.20, 0.24, -0.22, 0.18, 0.13, -0.10, 0.11],
                    [0.10, 0.34, 0.16, -0.24, 0.14, 0.12, -0.12],
                    [-0.10, -0.34, -0.16, 0.24, -0.14, -0.12, 0.12],
                    [0.24, 0.12, -0.26, 0.18, 0.16, -0.10, 0.12],
                    [-0.24, -0.12, 0.26, -0.18, -0.16, 0.10, -0.12],
                    [0.18, -0.32, -0.16, 0.24, -0.16, 0.14, 0.10],
                    [-0.18, 0.32, 0.16, -0.24, 0.16, -0.14, -0.10],
                ],
                dtype=float,
            )
            deltas = list(stage8_like)
        elif profile == "recording_showcase":
            # Five visually separated points for the final screen recording.
            # They intentionally move the EE lower and across left/right/front
            # regions of the learned workspace so the motion is obvious, while
            # keeping the target count short enough to record smoothly.
            stage8_like = np.array(
                [
                    [0.16, 0.30, -0.20, 0.22, 0.12, -0.12, 0.10],
                    [-0.24, -0.12, 0.26, -0.18, -0.16, 0.10, -0.12],
                    [-0.20, 0.24, -0.22, 0.18, 0.13, -0.10, 0.11],
                    [0.20, -0.24, 0.22, -0.18, -0.13, 0.10, -0.11],
                    [-0.18, 0.32, 0.16, -0.24, 0.16, -0.14, -0.10],
                ],
                dtype=float,
            )
            deltas = list(stage8_like)
    else:
        raise ValueError(f"Unknown target profile: {profile}")
    targets: list[TargetSpec] = []
    for idx, delta in enumerate(deltas[: max(0, int(max_targets))]):
        q_goal = clip_joint_configuration(q_reference + delta, joint_specs)
        targets.append(
            TargetSpec(
                name=f"reachable_fk_target_{idx}",
                goal_pose6=compute_ee_pose6(q_goal),
                source=f"default_q_delta_fk:{profile}",
                q_goal=q_goal,
            )
        )
    return targets


def load_targets_json(
    path: str | Path,
    *,
    q_reference: np.ndarray,
    joint_specs: Sequence[JointSpec],
) -> list[TargetSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        entries = payload.get("targets", [])
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ValueError("targets JSON must contain a list or {'targets': list}")

    targets: list[TargetSpec] = []
    for idx, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"target entry {idx} must be an object")
        name = str(raw.get("name", f"target_{idx}"))
        if "pose6" in raw:
            pose6 = np.asarray(raw["pose6"], dtype=float)
            if pose6.shape != (6,):
                raise ValueError(f"{name}: pose6 must have length 6")
            targets.append(TargetSpec(name=name, goal_pose6=pose6, source="pose6_json"))
        elif "q_goal" in raw:
            q_goal = clip_joint_configuration(np.asarray(raw["q_goal"], dtype=float), joint_specs)
            targets.append(TargetSpec(name=name, goal_pose6=compute_ee_pose6(q_goal), source="q_goal_json", q_goal=q_goal))
        elif "q_delta" in raw:
            q_goal = clip_joint_configuration(q_reference + np.asarray(raw["q_delta"], dtype=float), joint_specs)
            targets.append(TargetSpec(name=name, goal_pose6=compute_ee_pose6(q_goal), source="q_delta_json", q_goal=q_goal))
        else:
            raise ValueError(f"{name}: expected one of pose6, q_goal, q_delta")
    return targets


def _make_runtime(args: argparse.Namespace) -> RuntimeROS2Adapter:
    return RuntimeROS2Adapter.from_ros2(
        joint_names=list(JOINT_ORDER),
        trajectory_topic=args.trajectory_topic,
        joint_state_topic=args.joint_state_topic,
        command_duration_s=args.command_duration_s,
        settle_timeout_s=args.settle_timeout_s,
        initial_warmup_timeout_s=args.initial_warmup_timeout_s,
        use_action_primary=not args.topic_only,
        gz_visualize_target=True,
        gz_world_name=args.gz_world_name,
        gz_target_entity_name=args.gz_target_entity_name,
        gz_target_world_offset_xyz=tuple(args.gz_target_world_offset_xyz),
    )


def _handoff_ready(
    *,
    pos_error: float,
    ori_error: float,
    action_l2: float,
    dq_norm: float,
    handoff: Mapping[str, Any],
) -> bool:
    pos_threshold = float(handoff.get("pos_threshold_m", handoff.get("position_error_m", 0.002)))
    ori_threshold = float(handoff.get("ori_threshold_rad", handoff.get("orientation_error_rad", 0.02)))
    action_threshold = float(handoff.get("action_magnitude_threshold", handoff.get("action_magnitude", 0.10)))
    dq_threshold = float(handoff.get("dq_norm_threshold", handoff.get("dq_norm", 0.002)))
    return bool(
        pos_error <= pos_threshold
        and ori_error <= ori_threshold
        and action_l2 <= action_threshold
        and dq_norm <= dq_threshold
    )


def _strict_success(pos_error: float, ori_error: float, handoff: Mapping[str, Any] | None = None) -> bool:
    handoff = handoff or {}
    pos_threshold = float(handoff.get("strict_position_error_m", 0.005))
    ori_threshold = float(handoff.get("strict_orientation_error_rad", 0.05))
    return bool(pos_error <= pos_threshold and ori_error <= ori_threshold)


def _write_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _rollout_phase(
    *,
    runtime: RuntimeROS2Adapter,
    policy: Any,
    phase_cfg: PhaseRuntimeConfig,
    target: TargetSpec,
    joint_specs: Sequence[JointSpec],
    prev_q: np.ndarray,
    prev_action: np.ndarray,
    target_index: int,
    target_log_path: Path,
    handoff: Mapping[str, Any],
    stop_on_handoff: bool,
) -> dict[str, Any]:
    confirm_steps = int(handoff.get("confirm_steps", 2))
    consecutive_handoff_ready = 0
    dwell_success_steps = 0
    phase_rows = 0
    min_pos_error = math.inf
    min_ori_error = math.inf
    final_pos_error = math.inf
    final_ori_error = math.inf
    handoff_step: int | None = None

    q_last = np.asarray(prev_q, dtype=float)
    action_last = np.asarray(prev_action, dtype=float)

    for step in range(phase_cfg.episode_steps):
        q_before = runtime.read_q(timeout_s=5.0)
        dq = q_before - q_last
        pose_before = compute_ee_pose6(q_before)
        pos_before, ori_before = pose_error_norms(pose_before, target.goal_pose6)
        obs = build_observation(
            q=q_before,
            dq=dq,
            prev_action=action_last,
            current_pose6=pose_before,
            goal_pose6=target.goal_pose6,
            joint_specs=joint_specs,
            episode_progress=float(step) / max(1.0, float(phase_cfg.episode_steps - 1)),
            dwell_progress=float(dwell_success_steps) / 5.0,
            mode_index=phase_cfg.mode_index,
        )
        action = _predict_action(policy, obs)
        effective_delta_scale = effective_action_delta_scale(phase_cfg, pos_before)
        cmd_q = action_to_command_q(
            q=q_before,
            action=action,
            joint_specs=joint_specs,
            action_delta_scale=effective_delta_scale,
        )
        result = runtime.step(cmd_q)
        q_after = np.asarray(result["q_after"], dtype=float)
        pose_after = compute_ee_pose6(q_after)
        pos_after, ori_after = pose_error_norms(pose_after, target.goal_pose6)
        actual_dq = q_after - q_before
        action_l2 = _norm(action)
        dq_norm = _norm(actual_dq)
        tracking_error_l2 = _norm(q_after - cmd_q)

        strict_now = _strict_success(pos_after, ori_after, handoff)
        dwell_success_steps = dwell_success_steps + 1 if strict_now else 0
        ready_now = _handoff_ready(
            pos_error=pos_after,
            ori_error=ori_after,
            action_l2=action_l2,
            dq_norm=dq_norm,
            handoff=handoff,
        )
        consecutive_handoff_ready = consecutive_handoff_ready + 1 if ready_now else 0
        min_pos_error = min(min_pos_error, pos_after)
        min_ori_error = min(min_ori_error, ori_after)
        final_pos_error = pos_after
        final_ori_error = ori_after
        phase_rows += 1

        _write_jsonl_row(
            target_log_path,
            {
                "schema_version": "v5.phase3a.controlled_sim.step.v1",
                "target_index": target_index,
                "target_name": target.name,
                "phase": phase_cfg.role,
                "step": step,
                "q_before": q_before.tolist(),
                "q_after": q_after.tolist(),
                "cmd_q": cmd_q.tolist(),
                "goal_pose6": target.goal_pose6.tolist(),
                "pose_before": pose_before.tolist(),
                "pose_after": pose_after.tolist(),
                "position_error_before": pos_before,
                "orientation_error_before": ori_before,
                "position_error_after": pos_after,
                "orientation_error_after": ori_after,
                "policy_action": action.tolist(),
                "policy_action_l2": action_l2,
                "effective_action_delta_scale": effective_delta_scale,
                "cmd_delta_l2": float(result["cmd_delta_l2"]),
                "actual_joint_delta_l2": float(result["joint_delta_l2"]),
                "actual_dq_norm": dq_norm,
                "tracking_error_l2": tracking_error_l2,
                "handoff_ready": ready_now,
                "consecutive_handoff_ready": consecutive_handoff_ready,
                "strict_pose_success": strict_now,
                "strict_dwell_steps": dwell_success_steps,
                "execution_ok": bool(result["execution_ok"]),
                "fail_reason": str(result["fail_reason"]),
                "command_path": str(result["command_path"]),
                "timestamp_ns": int(result["timestamp_ns"]),
            },
        )

        q_last = q_after
        action_last = action
        if stop_on_handoff and consecutive_handoff_ready >= confirm_steps:
            handoff_step = step
            break

    return {
        "phase": phase_cfg.role,
        "steps": phase_rows,
        "handoff_step": handoff_step,
        "final_q": q_last.tolist(),
        "final_action": action_last.tolist(),
        "final_position_error": final_pos_error,
        "final_orientation_error": final_ori_error,
        "min_position_error": min_pos_error,
        "min_orientation_error": min_ori_error,
        "final_strict_dwell_steps": dwell_success_steps,
        "handoff_ready_confirmed": handoff_step is not None,
    }


def run_controlled_sim(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = Path(args.artifact_root)
    run_dir = artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    step_log_path = run_dir / "runtime_steps.jsonl"
    if step_log_path.exists():
        step_log_path.unlink()

    registry = load_phase3a_model_registry(args.registry)
    approach_cfg = phase_runtime_config(registry, role="APPROACH", override_steps=args.approach_steps)
    finisher_cfg = phase_runtime_config(registry, role="FINISHER", override_steps=args.finisher_steps)
    joint_specs = default_joint_specs()

    approach_policy = load_sb3_policy(approach_cfg.checkpoint, device=args.policy_device)
    finisher_policy = load_sb3_policy(finisher_cfg.checkpoint, device=args.policy_device)

    runtime = _make_runtime(args)
    try:
        q_initial = runtime.read_q(timeout_s=12.0)
        if args.targets_json:
            targets = load_targets_json(args.targets_json, q_reference=q_initial, joint_specs=joint_specs)
            targets = targets[: args.max_targets]
        else:
            targets = build_default_targets(
                q_reference=q_initial,
                joint_specs=joint_specs,
                max_targets=args.max_targets,
                profile=args.target_profile,
            )

        target_summaries: list[dict[str, Any]] = []
        for target_index, target in enumerate(targets):
            task_text = task_description(target.name, target_index, len(targets))
            print(
                f"[phase3a] target {target_index + 1}/{len(targets)} "
                f"{target.name} source={target.source}",
                flush=True,
            )
            print(f"[TASK] {task_text}", flush=True)
            print(
                f"[L2/RL] {target.name}: APPROACH policy starts for this subtask",
                flush=True,
            )
            visual = runtime.publish_ee_target_visual(target.goal_pose6)
            q_start = runtime.read_q(timeout_s=6.0)
            pose_start = compute_ee_pose6(q_start)
            start_pos_error, start_ori_error = pose_error_norms(pose_start, target.goal_pose6)
            print(
                f"[phase3a] target {target_index}: start_pos={start_pos_error:.4f}m "
                f"start_ori={start_ori_error:.4f}rad",
                flush=True,
            )

            approach = _rollout_phase(
                runtime=runtime,
                policy=approach_policy,
                phase_cfg=approach_cfg,
                target=target,
                joint_specs=joint_specs,
                prev_q=q_start,
                prev_action=np.zeros(len(joint_specs), dtype=float),
                target_index=target_index,
                target_log_path=step_log_path,
                handoff=registry.handoff,
                stop_on_handoff=True,
            )
            print(
                f"[phase3a] target {target_index}: approach_done "
                f"steps={approach['steps']} final_pos={float(approach['final_position_error']):.4f}m "
                f"final_ori={float(approach['final_orientation_error']):.4f}rad",
                flush=True,
            )
            print(
                f"[L2/RL] {target.name}: APPROACH finished; handing state to FINISHER",
                flush=True,
            )
            q_after_approach = np.asarray(approach["final_q"], dtype=float)
            finisher = _rollout_phase(
                runtime=runtime,
                policy=finisher_policy,
                phase_cfg=finisher_cfg,
                target=target,
                joint_specs=joint_specs,
                prev_q=q_after_approach,
                prev_action=np.asarray(approach["final_action"], dtype=float),
                target_index=target_index,
                target_log_path=step_log_path,
                handoff=registry.handoff,
                stop_on_handoff=False,
            )
            final_success = _strict_success(
                float(finisher["final_position_error"]),
                float(finisher["final_orientation_error"]),
                registry.handoff,
            )
            print(
                f"[phase3a] target {target_index}: finisher_done "
                f"steps={finisher['steps']} final_pos={float(finisher['final_position_error']):.4f}m "
                f"final_ori={float(finisher['final_orientation_error']):.4f}rad "
                f"success={final_success}",
                flush=True,
            )
            print(
                f"[L3/GZ] {target.name}: joint trajectory commands executed; "
                f"strict_success={final_success}",
                flush=True,
            )
            target_summaries.append(
                {
                    "target_index": target_index,
                    "target_name": target.name,
                    "target_source": target.source,
                    "q_goal": None if target.q_goal is None else target.q_goal.tolist(),
                    "goal_pose6": target.goal_pose6.tolist(),
                    "visualization": visual,
                    "start_position_error": start_pos_error,
                    "start_orientation_error": start_ori_error,
                    "approach": approach,
                    "finisher": finisher,
                    "final_success": final_success,
                }
            )

        summary = {
            "schema_version": "v5.phase3a.controlled_sim.summary.v1",
            "run_id": args.run_id,
            "artifact_root": str(artifact_root),
            "step_log_path": str(step_log_path),
            "registry": registry.to_dict(),
            "approach_checkpoint": repo_relative(approach_cfg.checkpoint),
            "finisher_checkpoint": repo_relative(finisher_cfg.checkpoint),
            "joint_order": list(JOINT_ORDER),
            "command_duration_s": float(args.command_duration_s),
            "settle_timeout_s": float(args.settle_timeout_s),
            "policy_device": args.policy_device,
            "target_count": len(targets),
            "success_rate": float(np.mean([x["final_success"] for x in target_summaries])) if target_summaries else 0.0,
            "final_position_error_mean": float(np.mean([x["finisher"]["final_position_error"] for x in target_summaries]))
            if target_summaries
            else math.nan,
            "final_orientation_error_mean": float(np.mean([x["finisher"]["final_orientation_error"] for x in target_summaries]))
            if target_summaries
            else math.nan,
            "handoff_confirmed_rate": float(np.mean([x["approach"]["handoff_ready_confirmed"] for x in target_summaries]))
            if target_summaries
            else 0.0,
            "targets": target_summaries,
        }
        summary_path = run_dir / "controlled_sim_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    finally:
        runtime.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled Phase 3A Approach->Finisher Gazebo validation")
    parser.add_argument("--registry", default=None, help="Override phase3a runtime model registry")
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts" / "v5" / "phase3a_controlled_sim"))
    parser.add_argument("--run-id", default=time.strftime("controlled_sim_%Y%m%d_%H%M%S"))
    parser.add_argument("--targets-json", default=None, help="Optional target list JSON")
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument(
        "--target-profile",
        choices=["smoke", "visible_workspace", "demo_showcase", "workspace_showcase", "recording_showcase"],
        default="smoke",
    )
    parser.add_argument("--approach-steps", type=int, default=None)
    parser.add_argument("--finisher-steps", type=int, default=None)
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--topic-only", action="store_true", help="Disable FollowJointTrajectory action path")
    parser.add_argument("--command-duration-s", type=float, default=0.35)
    parser.add_argument("--settle-timeout-s", type=float, default=1.4)
    parser.add_argument("--initial-warmup-timeout-s", type=float, default=4.0)
    parser.add_argument("--gz-world-name", default="empty")
    parser.add_argument("--gz-target-entity-name", default="phase3a_target_marker")
    parser.add_argument("--gz-target-world-offset-xyz", nargs=3, type=float, default=(0.0, 0.0, 1.04))
    parser.add_argument("--output-json", default=None, help="Optional extra copy of summary JSON")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_controlled_sim(args)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "target_count", "success_rate", "handoff_confirmed_rate", "final_position_error_mean", "final_orientation_error_mean", "step_log_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
