"""V5.1 end-to-end pipeline (real reward + minimal SAC)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .curriculum import CurriculumManager, resolve_stages
from .ee_fk import ee_pose6_from_q
from .gates import DEFAULT_GATE, GateEvaluator, write_gate_report
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig
from .pipeline_smoke import run_smoke
from .reward import RewardComposer, RewardTraceWriter
from .runtime_ros2 import RuntimeROS2Adapter


RuntimeFactory = Callable[..., RuntimeROS2Adapter]

_CONTROLLED_ACTION_DIM = 6
_OBS_DIM = 24
_NO_EFFECT_EPS = 1e-4
_NO_EFFECT_STREAK_LIMIT = 3
_HOME_Q = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
_EXTERNAL_TASK_LIBRARY_RELATIVE = Path(
    "external/ENPM662_Group4_FinalProject/src/kitchen_robot_controller/kitchen_robot_controller/task_library.py"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_ee_target_from_external_task(
    *,
    prop: str = "tray",
    src_idx: int = 2,
    dst_idx: int = 7,
    waypoint_index: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    external_file = _project_root() / _EXTERNAL_TASK_LIBRARY_RELATIVE
    spec = importlib.util.spec_from_file_location("v5_1_external_task_library", external_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load external task library spec: {external_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    task_lib = module.MoveTaskLibrary()
    trajectory = task_lib.move_from_to(prop=prop, src_idx=int(src_idx), dst_idx=int(dst_idx))
    if not trajectory:
        raise RuntimeError("external task trajectory is empty")

    idx = max(0, min(int(waypoint_index), len(trajectory) - 1))
    target_T = np.asarray(trajectory[idx], dtype=float)
    R = target_T[:3, :3]
    ee_target = np.array(
        [
            target_T[0, 3],
            target_T[1, 3],
            target_T[2, 3],
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)),
            np.arctan2(R[1, 0], R[0, 0]),
        ],
        dtype=float,
    )
    source = {
        "provider": "external_task_library.MoveTaskLibrary.move_from_to",
        "external_file": str(external_file),
        "prop": str(prop),
        "src_idx": int(src_idx),
        "dst_idx": int(dst_idx),
        "waypoint_index": int(idx),
        "trajectory_len": int(len(trajectory)),
    }
    return ee_target, source


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def _resolve_near_home_ee_target(
    *,
    home_q: np.ndarray,
    profile: str = "s0_bootstrap",
    pos_offset_min_m: float = 0.02,
    pos_offset_max_m: float = 0.05,
    ori_offset_min_deg: float = 5.0,
    ori_offset_max_deg: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    home_q = np.asarray(home_q, dtype=float)
    home_ee = _ee_pose_from_q(home_q)

    pos_mag = float(0.5 * (float(pos_offset_min_m) + float(pos_offset_max_m)))
    ori_mag_deg = float(0.5 * (float(ori_offset_min_deg) + float(ori_offset_max_deg)))
    ori_mag = float(np.deg2rad(ori_mag_deg))

    pos_dir = np.array([1.0, -0.5, 0.25], dtype=float)
    pos_dir = pos_dir / float(np.linalg.norm(pos_dir))
    ori_dir = np.array([1.0, -0.5, 0.25], dtype=float)
    ori_dir = ori_dir / float(np.linalg.norm(ori_dir))

    delta_pos = pos_dir * pos_mag
    delta_ori = ori_dir * ori_mag

    ee_target = home_ee.copy()
    ee_target[:3] = ee_target[:3] + delta_pos
    ee_target[3:6] = _wrap_to_pi(ee_target[3:6] + delta_ori)

    source = {
        "provider": "near_home_bootstrap",
        "profile": str(profile),
        "home_q": home_q.tolist(),
        "home_ee": home_ee.tolist(),
        "target_delta_pos": delta_pos.tolist(),
        "target_delta_ori": delta_ori.tolist(),
        "target_delta_pos_l2": float(np.linalg.norm(delta_pos)),
        "target_delta_ori_l2": float(np.linalg.norm(delta_ori)),
        "pos_offset_min_m": float(pos_offset_min_m),
        "pos_offset_max_m": float(pos_offset_max_m),
        "ori_offset_min_deg": float(ori_offset_min_deg),
        "ori_offset_max_deg": float(ori_offset_max_deg),
    }
    return ee_target, source


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _ee_pose_from_q(q: np.ndarray) -> np.ndarray:
    return ee_pose6_from_q(np.asarray(q, dtype=float))


def _ee_errors(ee_pose: np.ndarray, ee_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ee_pose = np.asarray(ee_pose, dtype=float)
    ee_target = np.asarray(ee_target, dtype=float)
    return ee_target[:3] - ee_pose[:3], ee_target[3:6] - ee_pose[3:6]


def _obs_from_state(q: np.ndarray, dq: np.ndarray, ee_pose_err: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    return np.concatenate([q, dq, ee_pose_err, prev_action], axis=0)


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _checkpoint_layout(artifact_root: Path) -> dict[str, Path]:
    train_root = artifact_root / "train"
    return {
        "latest": train_root / "checkpoint_latest.pt",
        "final": train_root / "checkpoint_final.pt",
    }


def _checkpoint_candidates(artifact_root: Path) -> list[Path]:
    layout = _checkpoint_layout(artifact_root)
    return [layout["latest"], layout["final"]]


def _load_agent_checkpoint(agent: Any, checkpoint_path: Path) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.q1.load_state_dict(payload["q1_state_dict"])
    agent.q2.load_state_dict(payload["q2_state_dict"])
    agent.q1_target.load_state_dict(payload["q1_target_state_dict"])
    agent.q2_target.load_state_dict(payload["q2_target_state_dict"])
    agent.actor_optim.load_state_dict(payload["actor_optim_state_dict"])
    agent.q1_optim.load_state_dict(payload["q1_optim_state_dict"])
    agent.q2_optim.load_state_dict(payload["q2_optim_state_dict"])
    agent.log_alpha = torch.tensor(
        float(payload.get("log_alpha", np.log(max(float(payload.get("alpha", 0.2)), 1e-8)))),
        dtype=torch.float32,
        device=agent.device,
        requires_grad=True,
    )
    agent.alpha_optim = torch.optim.Adam([agent.log_alpha], lr=agent.cfg.lr_alpha)
    if "alpha_optim_state_dict" in payload:
        agent.alpha_optim.load_state_dict(payload["alpha_optim_state_dict"])

    agent.env_steps_collected = int(payload.get("env_steps_collected", 0))
    agent.updates_applied = int(payload.get("updates_applied", 0))
    agent.batch_draw_count = int(payload.get("batch_draw_count", 0))
    agent.actor_update_count = int(payload.get("actor_update_count", 0))
    agent.critic_update_count = int(payload.get("critic_update_count", 0))
    agent.alpha_update_count = int(payload.get("alpha_update_count", 0))
    agent.last_actor_hash = payload.get("last_actor_hash")
    agent.last_critic_hash = payload.get("last_critic_hash")


def _save_agent_checkpoint(agent: Any, checkpoint_path: Path, run_id: str) -> str:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp_ns": time.time_ns(),
        "actor_state_dict": agent.actor.state_dict(),
        "q1_state_dict": agent.q1.state_dict(),
        "q2_state_dict": agent.q2.state_dict(),
        "q1_target_state_dict": agent.q1_target.state_dict(),
        "q2_target_state_dict": agent.q2_target.state_dict(),
        "actor_optim_state_dict": agent.actor_optim.state_dict(),
        "q1_optim_state_dict": agent.q1_optim.state_dict(),
        "q2_optim_state_dict": agent.q2_optim.state_dict(),
        "alpha_optim_state_dict": agent.alpha_optim.state_dict(),
        "log_alpha": float(agent.log_alpha.detach().cpu().item()),
        "alpha": float(agent.alpha),
        "env_steps_collected": int(agent.env_steps_collected),
        "updates_applied": int(agent.updates_applied),
        "batch_draw_count": int(agent.batch_draw_count),
        "actor_update_count": int(agent.actor_update_count),
        "critic_update_count": int(agent.critic_update_count),
        "alpha_update_count": int(agent.alpha_update_count),
        "last_actor_hash": agent.last_actor_hash,
        "last_critic_hash": agent.last_critic_hash,
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def _controlled_joint_indices(runtime_joint_names: list[str]) -> list[int]:
    indices = [i for i, name in enumerate(runtime_joint_names) if name.lower() != "rack_joint"]
    if len(indices) != _CONTROLLED_ACTION_DIM:
        raise ValueError(
            "runtime_joint_names must resolve to exactly 6 controllable joints "
            f"(excluding Rack_joint); got {len(indices)} from {runtime_joint_names}"
        )
    return indices


def _expand_cmd_q(q_before_full: np.ndarray, controlled_indices: list[int], q_des_controlled: np.ndarray) -> np.ndarray:
    cmd_full = np.asarray(q_before_full, dtype=float).copy()
    cmd_full[np.asarray(controlled_indices, dtype=int)] = np.asarray(q_des_controlled, dtype=float)
    return cmd_full


def _reset_episode_home(
    runtime: RuntimeROS2Adapter,
    controlled_indices: list[int],
    home_q: np.ndarray,
    reset_near_home_eps: float,
) -> dict[str, Any]:
    q_before_full = runtime.read_q()
    controlled_idx_np = np.asarray(controlled_indices, dtype=int)
    q_before_controlled = np.asarray(q_before_full, dtype=float)[controlled_idx_np]
    home_q = np.asarray(home_q, dtype=float)
    near_home_l2 = float(np.linalg.norm(q_before_controlled - home_q))
    reset_skipped_near_home = bool(near_home_l2 < float(reset_near_home_eps))

    if reset_skipped_near_home:
        return {
            "accepted": True,
            "result_status": "success",
            "execution_ok": True,
            "fail_reason": "none",
            "command_path": "skipped_near_home",
            "home_q": home_q.tolist(),
            "q_after": np.asarray(q_before_full, dtype=float).tolist(),
            "q_before": np.asarray(q_before_full, dtype=float).tolist(),
            "reset_skipped_near_home": True,
            "reset_near_home_l2": near_home_l2,
            "runtime": None,
        }

    cmd_q_full = _expand_cmd_q(q_before_full=q_before_full, controlled_indices=controlled_indices, q_des_controlled=home_q)
    out = runtime.step(cmd_q_full)
    return {
        "accepted": bool(out.get("accepted", False)),
        "result_status": str(out.get("result_status", "fail")),
        "execution_ok": bool(out.get("execution_ok", False)),
        "fail_reason": str(out.get("fail_reason", "unknown")),
        "command_path": str(out.get("command_path", "unknown")),
        "home_q": home_q.tolist(),
        "q_after": out.get("q_after", []),
        "q_before": out.get("q_before", np.asarray(q_before_full, dtype=float).tolist()),
        "reset_skipped_near_home": False,
        "reset_near_home_l2": near_home_l2,
        "runtime": out,
    }


def _run_episode_gz(
    *,
    ep_id: str,
    ep_index: int,
    step_count: int,
    logs_root: Path,
    runtime: RuntimeROS2Adapter,
    ee_target: np.ndarray,
    controlled_indices: list[int],
    policy_fn: Callable[[np.ndarray], tuple[np.ndarray, str]],
    action_limit: float,
    no_effect_epsilon: float = _NO_EFFECT_EPS,
    no_effect_streak_limit: int = _NO_EFFECT_STREAK_LIMIT,
) -> dict[str, Any]:
    ts0 = time.time_ns()
    executor = L3DeterministicExecutor(
        L3ExecutorConfig(dt=0.1, delta_q_limit=(float(action_limit),) * _CONTROLLED_ACTION_DIM)
    )

    l1_path = logs_root / "l1" / f"{ep_id}.jsonl"
    l2_path = logs_root / "l2" / f"{ep_id}.jsonl"
    l3_path = logs_root / "l3" / f"{ep_id}.jsonl"

    prev_q_des: np.ndarray | None = None
    trace_steps: list[dict[str, Any]] = []
    controlled_idx_np = np.asarray(controlled_indices, dtype=int)
    no_effect_streak = 0
    prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
    prev_q: np.ndarray | None = None

    for step in range(max(1, int(step_count))):
        now_ns = ts0 + step * 100_000_000
        q_before_full = runtime.read_q()
        q_before = q_before_full[controlled_idx_np]
        ee_pose_before = _ee_pose_from_q(q_before_full)
        ee_pos_err_prev, ee_ori_err_prev = _ee_errors(ee_pose_before, ee_target)
        goal_error_prev = RewardComposer.ee_error_norm(ee_pos_err_prev, ee_ori_err_prev)
        dq = (q_before - prev_q) if prev_q is not None else np.zeros_like(q_before)
        obs = _obs_from_state(q=q_before, dq=dq, ee_pose_err=np.concatenate([ee_pos_err_prev, ee_ori_err_prev]), prev_action=prev_action)
        action_raw, policy_name = policy_fn(obs)
        prev_q = q_before.copy()

        l1_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "observation": {"q": q_before.tolist(), "ee_target": ee_target.tolist(), "goal_error_l2": goal_error_prev},
        }
        _jsonl_append(l1_path, l1_payload)

        exec_out = executor.compute_q_des(q_current=q_before, delta_q_cmd=action_raw, prev_q_des=prev_q_des)
        prev_q_des = exec_out.q_des
        saturation = bool(np.any(np.abs(exec_out.clamped_delta_q - exec_out.requested_delta_q) > 1e-12))

        l2_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "policy": policy_name,
            "action_raw": exec_out.requested_delta_q.tolist(),
            "action_clamped": exec_out.clamped_delta_q.tolist(),
            "projection_applied": bool(exec_out.projection_applied),
            "saturated": saturation,
        }
        _jsonl_append(l2_path, l2_payload)

        cmd_q_full = _expand_cmd_q(q_before_full=q_before_full, controlled_indices=controlled_indices, q_des_controlled=exec_out.q_des)
        rt = runtime.step(cmd_q_full)
        q_after_full = np.asarray(rt["q_after"], dtype=float)
        q_after = q_after_full[controlled_idx_np]
        ee_pose_after = _ee_pose_from_q(q_after_full)
        ee_pos_err_next, ee_ori_err_next = _ee_errors(ee_pose_after, ee_target)
        goal_error_next = RewardComposer.ee_error_norm(ee_pos_err_next, ee_ori_err_next)

        rt_joint_delta_l2 = float(rt["joint_delta_l2"])
        rt_no_effect = rt.get("no_effect")
        no_effect = bool(rt_no_effect) if rt_no_effect is not None else (rt_joint_delta_l2 < float(no_effect_epsilon))
        no_effect_streak = (no_effect_streak + 1) if no_effect else 0

        l3_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "cmd_q": rt["cmd_q"],
            "q_before": rt["q_before"],
            "q_after": rt["q_after"],
            "joint_delta_l2": rt_joint_delta_l2,
            "goal_error_l2": goal_error_next,
            "no_effect": bool(no_effect),
            "no_effect_streak": int(no_effect_streak),
            "accepted": bool(rt.get("accepted", True)),
            "result_status": rt.get("result_status", "success"),
            "execution_ok": bool(rt.get("execution_ok", True)),
            "fail_reason": rt.get("fail_reason", "none"),
        }
        _jsonl_append(l3_path, l3_payload)

        intervention = "none"
        if no_effect_streak >= int(no_effect_streak_limit) and float(np.linalg.norm(ee_pos_err_next)) >= 0.08:
            intervention = "no_effect"

        trace_steps.append(
            {
                "step": int(step),
                "obs_q": q_before.tolist(),
                "q_after": q_after.tolist(),
                "ee_pose": ee_pose_before.tolist(),
                "ee_target": ee_target.tolist(),
                "ee_pos_err": ee_pos_err_next.tolist(),
                "ee_ori_err": ee_ori_err_next.tolist(),
                "action_raw": exec_out.requested_delta_q.tolist(),
                "action_clamped": exec_out.clamped_delta_q.tolist(),
                "goal_error_prev": goal_error_prev,
                "goal_error_next": goal_error_next,
                "intervention": intervention,
                "projection_applied": bool(exec_out.projection_applied),
                "saturated": saturation,
                "no_effect": bool(no_effect),
                "no_effect_streak": int(no_effect_streak),
                "runtime": rt,
            }
        )

        prev_action = np.asarray(exec_out.requested_delta_q, dtype=float)

        if intervention == "no_effect":
            break

    return {
        "l1": str(l1_path),
        "l2": str(l2_path),
        "l3": str(l3_path),
        "trace_steps": trace_steps,
        "final_goal_error": float(trace_steps[-1]["goal_error_next"]) if trace_steps else 0.0,
    }


def run_pipeline_e2e(
    run_id: str,
    episodes: int,
    steps_per_episode: int,
    artifact_root: Path,
    enforce_gates: bool = False,
    policy_mode: str = "sac_torch",
    sac_seed: int = 0,
    stage_profile: str = "default",
    runtime_mode: str = "smoke",
    runtime_factory: RuntimeFactory | None = None,
    runtime_joint_names: list[str] | None = None,
    trajectory_topic: str = "/arm_controller/joint_trajectory",
    joint_state_topic: str = "/joint_states",
    ee_pos_success_threshold: float = 0.08,
    ee_ori_success_threshold: float = 0.12,
    reset_near_home_eps: float = 1e-4,
    external_task_prop: str = "tray",
    external_task_src_idx: int = 2,
    external_task_dst_idx: int = 7,
    external_task_waypoint_index: int = 2,
    target_mode: str = "auto",
    near_home_profile: str = "s0_bootstrap",
    near_home_pos_offset_min_m: float = 0.02,
    near_home_pos_offset_max_m: float = 0.05,
    near_home_ori_offset_min_deg: float = 5.0,
    near_home_ori_offset_max_deg: float = 10.0,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    reward_trace_path = artifact_root / "reward_trace.jsonl"
    reward_trace = RewardTraceWriter(reward_trace_path)
    reward_composer = RewardComposer()
    episode_reward_summary_path = artifact_root / "episode_reward_summary.jsonl"
    episode_reward_summary_path.write_text("", encoding="utf-8")

    runtime_trace_path = artifact_root / "runtime_trace.jsonl"
    runtime_trace_path.write_text("", encoding="utf-8")

    curriculum = CurriculumManager(stages=resolve_stages(stage_profile))
    gate_eval = GateEvaluator(DEFAULT_GATE)

    if policy_mode != "sac_torch":
        raise ValueError("V5.1 single-path only supports policy_mode=sac_torch")
    if runtime_mode not in {"smoke", "gz"}:
        raise ValueError("runtime_mode must be one of: smoke|gz")
    if target_mode not in {"auto", "external", "near_home"}:
        raise ValueError("target_mode must be one of: auto|external|near_home")

    from .sac_torch import SACTorchAgent, SACTorchConfig

    agent: Any = SACTorchAgent(
        SACTorchConfig(obs_dim=_OBS_DIM, action_dim=_CONTROLLED_ACTION_DIM),
        seed=sac_seed,
    )

    loaded_from_checkpoint = False
    loaded_checkpoint_path: str | None = None
    for checkpoint_candidate in _checkpoint_candidates(artifact_root):
        if checkpoint_candidate.exists():
            _load_agent_checkpoint(agent, checkpoint_candidate)
            loaded_from_checkpoint = True
            loaded_checkpoint_path = str(checkpoint_candidate)
            break

    runtime = None
    runtime_controlled_indices: list[int] | None = None
    if runtime_mode == "gz":
        if not runtime_joint_names:
            raise ValueError("runtime_joint_names is required for runtime_mode=gz")
        runtime_controlled_indices = _controlled_joint_indices(runtime_joint_names)
        factory = runtime_factory or RuntimeROS2Adapter.from_ros2
        runtime = factory(
            joint_names=runtime_joint_names,
            trajectory_topic=trajectory_topic,
            joint_state_topic=joint_state_topic,
        )

    external_ee_target, external_ee_target_source = _resolve_ee_target_from_external_task(
        prop=external_task_prop,
        src_idx=external_task_src_idx,
        dst_idx=external_task_dst_idx,
        waypoint_index=external_task_waypoint_index,
    )

    episodes_requested = max(1, int(episodes))
    successes = 0
    interventions = 0
    episode_outputs: list[dict[str, Any]] = []
    success_series: list[float] = []
    intervention_series: list[float] = []
    expected_log_lines_per_layer = 0
    reset_failures = 0
    reward_totals: list[float] = []
    train_metrics: list[dict[str, float]] = []
    episode_joint_delta_summary: list[dict[str, Any]] = []
    reset_failure_reasons: list[str] = []

    try:
        for ep in range(episodes_requested):
            stage = curriculum.current_stage
            if int(stage.controlled_dofs) != _CONTROLLED_ACTION_DIM:
                raise ValueError(f"unsupported controlled_dofs={stage.controlled_dofs}, expected {_CONTROLLED_ACTION_DIM}")
            ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
            step_count = min(int(steps_per_episode), stage.step_budget)

            resolved_target_mode = target_mode
            if resolved_target_mode == "auto":
                resolved_target_mode = "near_home" if stage.name == "S0_B" else "external"

            home_q_for_target = _HOME_Q.copy()
            if runtime_mode == "gz" and runtime is not None and runtime_controlled_indices is not None:
                q_home_full = runtime.read_q()
                home_q_for_target = np.asarray(q_home_full, dtype=float)[np.asarray(runtime_controlled_indices, dtype=int)]

            if resolved_target_mode == "near_home":
                ee_target, ee_target_source = _resolve_near_home_ee_target(
                    home_q=home_q_for_target,
                    profile=near_home_profile,
                    pos_offset_min_m=near_home_pos_offset_min_m,
                    pos_offset_max_m=near_home_pos_offset_max_m,
                    ori_offset_min_deg=near_home_ori_offset_min_deg,
                    ori_offset_max_deg=near_home_ori_offset_max_deg,
                )
            else:
                ee_target = np.asarray(external_ee_target, dtype=float).copy()
                ee_target_source = dict(external_ee_target_source)

            def _policy_fn(obs: np.ndarray) -> tuple[np.ndarray, str]:
                return agent.act(obs, stochastic=True), policy_mode

            reset_info = {"status": "skipped", "accepted": True, "execution_ok": True, "result_status": "skipped"}
            try:
                if runtime_mode == "smoke":
                    def _smoke_policy_fn(q: np.ndarray, target_q: np.ndarray) -> tuple[np.ndarray, str]:
                        ee_pose = _ee_pose_from_q(q)
                        ee_pos_err, ee_ori_err = _ee_errors(ee_pose, ee_target)
                        obs = _obs_from_state(q=q, dq=np.zeros_like(q), ee_pose_err=np.concatenate([ee_pos_err, ee_ori_err]), prev_action=np.zeros_like(q))
                        return _policy_fn(obs)

                    logs = run_smoke(
                        run_id=ep_id,
                        steps=step_count,
                        log_root=logs_root,
                        episode=ep,
                        policy_fn=_smoke_policy_fn,
                        action_limit=float(stage.action_limit),
                        target_q=ee_target,
                    )
                else:
                    reset_info = _reset_episode_home(
                        runtime=runtime,
                        controlled_indices=runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)),
                        home_q=_HOME_Q,
                        reset_near_home_eps=reset_near_home_eps,
                    )
                    _jsonl_append(runtime_trace_path, {"run_id": run_id, "episode": ep, "step": -1, "event": "reset", **reset_info})
                    if not (reset_info["accepted"] and reset_info["execution_ok"]):
                        reset_failures += 1
                        reset_failure_reasons.append(
                            f"ep={ep}:reset:{reset_info.get('fail_reason','unknown')}"
                        )
                        _jsonl_append(
                            episode_reward_summary_path,
                            {
                                "run_id": run_id,
                                "episode": ep,
                                "episode_id": ep_id,
                                "reset_result": reset_info,
                                "reset_fail": True,
                                "reset_skipped_near_home": bool(reset_info.get("reset_skipped_near_home", False)),
                            },
                        )
                        break

                    if resolved_target_mode == "near_home":
                        q_after_reset = np.asarray(reset_info.get("q_after", home_q_for_target.tolist()), dtype=float)
                        home_q_after_reset = q_after_reset[np.asarray(runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)), dtype=int)]
                        ee_target, ee_target_source = _resolve_near_home_ee_target(
                            home_q=home_q_after_reset,
                            profile=near_home_profile,
                            pos_offset_min_m=near_home_pos_offset_min_m,
                            pos_offset_max_m=near_home_pos_offset_max_m,
                            ori_offset_min_deg=near_home_ori_offset_min_deg,
                            ori_offset_max_deg=near_home_ori_offset_max_deg,
                        )

                    last_err: Exception | None = None
                    for _attempt in range(2):
                        try:
                            logs = _run_episode_gz(
                                ep_id=ep_id,
                                ep_index=ep,
                                step_count=step_count,
                                logs_root=logs_root,
                                runtime=runtime,
                                ee_target=ee_target,
                                controlled_indices=runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)),
                                policy_fn=_policy_fn,
                                action_limit=float(stage.action_limit),
                            )
                            last_err = None
                            break
                        except Exception as e:  # transient startup/readback races
                            last_err = e
                            time.sleep(0.25)
                    if last_err is not None:
                        raise last_err
            except Exception as e:
                reset_failures += 1
                reset_failure_reasons.append(f"ep={ep}:{type(e).__name__}:{e}")
                break

            expected_log_lines_per_layer += max(1, int(step_count))

            episode_return = 0.0
            ep_intervention = 0
            prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
            trace_steps = logs.get("trace_steps", [])
            ep_component_sums: dict[str, float] = {
                "progress": 0.0,
                "action": 0.0,
                "jerk": 0.0,
                "intervention": 0.0,
                "clamp_or_projection": 0.0,
                "stall": 0.0,
                "ee_small_motion_penalty": 0.0,
                "timeout_or_reset": 0.0,
                "success_bonus": 0.0,
                "reward_total": 0.0,
            }

            ep_q_before = np.asarray(trace_steps[0]["runtime"]["q_before"], dtype=float) if (trace_steps and runtime_mode == "gz") else None
            ep_q_after = np.asarray(trace_steps[-1]["runtime"]["q_after"], dtype=float) if (trace_steps and runtime_mode == "gz") else None
            terminal_done = False
            terminal_done_reason = "none"
            terminal_step: int | None = None

            for idx, step in enumerate(trace_steps):
                action = np.asarray(step["action_raw"], dtype=float)
                step_ee_target = np.asarray(step.get("ee_target", ee_target.tolist()), dtype=float)
                step_ee_pose = np.asarray(step.get("ee_pose", np.zeros(6, dtype=float)), dtype=float)
                prev_ee_pos_err = np.asarray(step_ee_target[:3], dtype=float) - np.asarray(step_ee_pose[:3], dtype=float)
                prev_ee_ori_err = np.asarray(step_ee_target[3:6], dtype=float) - np.asarray(step_ee_pose[3:6], dtype=float)
                curr_ee_pos_err = np.asarray(step.get("ee_pos_err", prev_ee_pos_err), dtype=float)
                curr_ee_ori_err = np.asarray(step.get("ee_ori_err", prev_ee_ori_err), dtype=float)
                prev_error = float(step.get("goal_error_prev", RewardComposer.ee_error_norm(prev_ee_pos_err, prev_ee_ori_err)))
                curr_error = float(step.get("goal_error_next", RewardComposer.ee_error_norm(curr_ee_pos_err, curr_ee_ori_err)))
                ee_step_dpos = float(np.linalg.norm(curr_ee_pos_err - prev_ee_pos_err))
                ee_step_dori = float(np.linalg.norm(curr_ee_ori_err - prev_ee_ori_err))
                intervention_now = step["intervention"] != "none"
                clamp_or_projection = bool(step["saturated"] or step["projection_applied"])
                execution_ok = bool(step.get("runtime", {}).get("execution_ok", True))

                done = idx == len(trace_steps) - 1
                done_reason = "running"
                if not execution_ok:
                    done = True
                    done_reason = "execution_fail"
                elif done:
                    if step["intervention"] == "no_effect":
                        done_reason = "no_effect"
                    else:
                        done_reason = "success" if (float(np.linalg.norm(curr_ee_pos_err)) < float(ee_pos_success_threshold) and float(np.linalg.norm(curr_ee_ori_err)) < float(ee_ori_success_threshold)) else "timeout"

                if done:
                    terminal_done = True
                    terminal_done_reason = done_reason
                    terminal_step = int(step["step"])

                q_before_arr = np.asarray(step.get("obs_q", []), dtype=float)
                q_after_arr = np.asarray(step.get("q_after", []), dtype=float)
                q_before_for_stall = q_before_arr if q_before_arr.size > 0 and q_before_arr.shape == q_after_arr.shape else None
                q_after_for_stall = q_after_arr if q_before_for_stall is not None else None

                terms = reward_composer.compute(
                    prev_ee_pos_err=prev_ee_pos_err,
                    prev_ee_ori_err=prev_ee_ori_err,
                    curr_ee_pos_err=curr_ee_pos_err,
                    curr_ee_ori_err=curr_ee_ori_err,
                    action=action,
                    prev_action=prev_action,
                    intervention=intervention_now,
                    clamp_or_projection=clamp_or_projection,
                    done=done,
                    done_reason=done_reason,
                    q_before=q_before_for_stall,
                    q_after=q_after_for_stall,
                    effect_ratio=(step.get("runtime", {}) or {}).get("effect_ratio"),
                )
                prev_action = action
                terms_dict = terms.to_dict()
                episode_return += terms.reward_total
                reward_totals.append(terms.reward_total)
                for k in ep_component_sums:
                    ep_component_sums[k] += float(terms_dict[k])

                reward_trace.append(
                    {
                        "run_id": run_id,
                        "episode": ep,
                        "episode_id": ep_id,
                        "step": int(step["step"]),
                        "policy_mode": policy_mode,
                        "runtime_mode": runtime_mode,
                        "done": done,
                        "done_reason": done_reason,
                        "goal_error_prev": prev_error,
                        "goal_error_next": curr_error,
                        "ee_pose": step.get("ee_pose"),
                        "ee_target": step.get("ee_target"),
                        "ee_pos_err": step.get("ee_pos_err"),
                        "ee_ori_err": step.get("ee_ori_err"),
                        "ee_step_dpos": ee_step_dpos,
                        "ee_step_dori": ee_step_dori,
                        "ee_small_motion_penalty": float(terms.ee_small_motion_penalty),
                        "reward_total": float(terms.reward_total),
                        "components": terms_dict,
                    }
                )

                if runtime_mode == "gz":
                    _jsonl_append(
                        runtime_trace_path,
                        {
                            "run_id": run_id,
                            "episode": ep,
                            "step": int(step["step"]),
                            "cmd_q": step["runtime"]["cmd_q"],
                            "readback_q_before": step["runtime"]["q_before"],
                            "readback_q_after": step["runtime"]["q_after"],
                            "joint_delta": step["runtime"]["joint_delta"],
                            "joint_delta_l2": step["runtime"]["joint_delta_l2"],
                            "cmd_delta_l2": step["runtime"].get("cmd_delta_l2"),
                            "effect_ratio": step["runtime"].get("effect_ratio"),
                            "frame_before_stamp_ns": step["runtime"].get("frame_before_stamp_ns"),
                            "frame_after_stamp_ns": step["runtime"].get("frame_after_stamp_ns"),
                            "goal_error_prev": prev_error,
                            "goal_error_next": curr_error,
                            "ee_pose": step.get("ee_pose"),
                            "ee_target": step.get("ee_target"),
                            "ee_pos_err": step.get("ee_pos_err"),
                            "ee_ori_err": step.get("ee_ori_err"),
                            "no_effect": bool(step.get("no_effect", False)),
                            "no_effect_reason": step["runtime"].get("no_effect_reason", "none"),
                            "no_effect_streak": int(step.get("no_effect_streak", 0)),
                            "skipped_publish": bool(step["runtime"].get("skipped_publish", False)),
                            "accepted": bool(step["runtime"].get("accepted", True)),
                            "result_status": step["runtime"].get("result_status", "success"),
                            "execution_ok": bool(step["runtime"].get("execution_ok", True)),
                            "fail_reason": step["runtime"].get("fail_reason", "none"),
                            "command_path": step["runtime"].get("command_path", "topic_fallback"),
                            "action_status": step["runtime"].get("action_status"),
                            "action_error_code": step["runtime"].get("action_error_code"),
                            "timestamp_ns": step["runtime"]["timestamp_ns"],
                        },
                    )

                if intervention_now:
                    ep_intervention = 1

                q_now = np.asarray(step["obs_q"], dtype=float)
                q_next = np.asarray(step["q_after"], dtype=float) if runtime_mode == "gz" else (q_now + np.asarray(step["action_clamped"], dtype=float))
                dq_now = np.zeros_like(q_now) if idx == 0 else (q_now - np.asarray(trace_steps[idx - 1]["obs_q"], dtype=float))
                dq_next = q_next - q_now
                ee_pose_err_now = np.concatenate([prev_ee_pos_err, prev_ee_ori_err])
                ee_pose_err_next = np.concatenate([curr_ee_pos_err, curr_ee_ori_err])
                obs = _obs_from_state(q=q_now, dq=dq_now, ee_pose_err=ee_pose_err_now, prev_action=prev_action)
                next_obs = _obs_from_state(q=q_next, dq=dq_next, ee_pose_err=ee_pose_err_next, prev_action=action)
                agent.remember(obs, action, terms.reward_total, next_obs, done)
                train_out = agent.train_step()
                if train_out is not None:
                    train_metrics.append(train_out)

                if done_reason == "execution_fail":
                    ep_intervention = 1
                    break

            final_error = float(logs.get("final_goal_error", 1.0))
            if trace_steps:
                last = trace_steps[-1]
                pos_ok = float(np.linalg.norm(np.asarray(last.get("ee_pos_err", [9, 9, 9]), dtype=float))) < float(ee_pos_success_threshold)
                ori_ok = float(np.linalg.norm(np.asarray(last.get("ee_ori_err", [9, 9, 9]), dtype=float))) < float(ee_ori_success_threshold)
                ep_success = 1 if (pos_ok and ori_ok) else 0
            else:
                ep_success = 0

            if runtime_mode == "gz" and ep_q_before is not None and ep_q_after is not None:
                delta = ep_q_after - ep_q_before
                episode_joint_delta_summary.append(
                    {
                        "episode": ep,
                        "before_q": ep_q_before.tolist(),
                        "after_q": ep_q_after.tolist(),
                        "delta_q": delta.tolist(),
                        "delta_q_l2": float(np.linalg.norm(delta)),
                    }
                )

            record = curriculum.record_episode(success_rate=float(ep_success))
            successes += ep_success
            interventions += ep_intervention
            success_series.append(float(ep_success))
            intervention_series.append(float(ep_intervention))

            step_n = max(1, len(trace_steps))
            ep_summary = {
                "run_id": run_id,
                "episode": ep,
                "episode_id": ep_id,
                "stage": record.stage_name,
                "component_sums": ep_component_sums,
                "component_means": {k: float(v) / float(step_n) for k, v in ep_component_sums.items()},
                "total_reward": float(ep_component_sums["reward_total"]),
                "success_count": int(ep_success),
                "intervention_count": int(ep_intervention),
                "no_effect_count": int(1 if trace_steps and trace_steps[-1]["intervention"] == "no_effect" else 0),
                "target_mode": resolved_target_mode,
                "home_ee": ee_target_source.get("home_ee"),
                "ee_target": ee_target.tolist(),
                "target_delta_pos": ee_target_source.get("target_delta_pos"),
                "target_delta_ori": ee_target_source.get("target_delta_ori"),
                "reset_result": reset_info,
                "reset_skipped_near_home": bool(reset_info.get("reset_skipped_near_home", False)),
                "terminal": bool(terminal_done),
                "done_reason": terminal_done_reason,
                "terminal_step": terminal_step,
            }
            _jsonl_append(episode_reward_summary_path, ep_summary)

            episode_outputs.append(
                {
                    "episode": ep,
                    "run_id": ep_id,
                    "stage": record.stage_name,
                    "success_rate": float(ep_success),
                    "promoted": record.promoted,
                    "logs": {k: v for k, v in logs.items() if k in {"l1", "l2", "l3"}},
                    "has_intervention": bool(ep_intervention),
                    "episode_return": float(episode_return),
                    "final_goal_error": final_error,
                    "policy_mode": policy_mode,
                    "runtime_mode": runtime_mode,
                    "target_mode": resolved_target_mode,
                    "home_ee": ee_target_source.get("home_ee"),
                    "ee_target": ee_target.tolist(),
                    "target_delta_pos": ee_target_source.get("target_delta_pos"),
                    "target_delta_ori": ee_target_source.get("target_delta_ori"),
                    "ee_target_source": ee_target_source,
                    "reset_result": reset_info,
                    "terminal": bool(terminal_done),
                    "done_reason": terminal_done_reason,
                    "terminal_step": terminal_step,
                }
            )
    finally:
        if runtime is not None:
            runtime.close()

    metrics = {
        "episodes_requested": episodes_requested,
        "episodes_completed": len(episode_outputs),
        "success_rate": _safe_rate(successes, len(episode_outputs)),
        "intervention_rate": _safe_rate(interventions, len(episode_outputs)),
        "success_rate_first": success_series[0] if success_series else 0.0,
        "success_rate_last": success_series[-1] if success_series else 0.0,
        "intervention_rate_first": intervention_series[0] if intervention_series else 0.0,
        "intervention_rate_last": intervention_series[-1] if intervention_series else 0.0,
        "reset_failures": reset_failures,
        "log_lines_expected_per_layer": expected_log_lines_per_layer,
        "reward_mean": float(np.mean(reward_totals)) if reward_totals else 0.0,
        "reward_std": float(np.std(reward_totals)) if reward_totals else 0.0,
        "reward_min": float(np.min(reward_totals)) if reward_totals else 0.0,
        "reward_max": float(np.max(reward_totals)) if reward_totals else 0.0,
    }
    if reset_failure_reasons:
        metrics["reset_failure_reasons"] = reset_failure_reasons[-5:]

    learning_effective = False
    ineffective_reasons: list[str] = []
    if train_metrics:
        metrics["train_actor_loss"] = float(np.mean([m["actor_loss"] for m in train_metrics]))
        metrics["train_critic_loss"] = float(np.mean([m["critic_loss"] for m in train_metrics]))
        metrics["train_alpha"] = float(train_metrics[-1]["alpha"])
        metrics["env_steps_collected"] = float(train_metrics[-1].get("env_steps_collected", 0.0))
        metrics["updates_applied"] = float(train_metrics[-1].get("updates_applied", 0.0))
        metrics["batch_draw_count"] = float(train_metrics[-1].get("batch_draw_count", 0.0))
        metrics["actor_update_count"] = float(train_metrics[-1].get("actor_update_count", 0.0))
        metrics["critic_update_count"] = float(train_metrics[-1].get("critic_update_count", 0.0))
        metrics["alpha_update_count"] = float(train_metrics[-1].get("alpha_update_count", 0.0))
        metrics["gradient_norm_actor"] = float(train_metrics[-1].get("gradient_norm_actor", 0.0))
        metrics["gradient_norm_critic"] = float(train_metrics[-1].get("gradient_norm_critic", 0.0))

        actor_hashes = [m.get("param_hash_actor") for m in train_metrics if m.get("param_hash_actor")]
        critic_hashes = [m.get("param_hash_critic") for m in train_metrics if m.get("param_hash_critic")]
        actor_hash_changed = len(set(actor_hashes)) > 1
        critic_hash_changed = len(set(critic_hashes)) > 1
        metrics["param_hash_actor_changed"] = actor_hash_changed
        metrics["param_hash_critic_changed"] = critic_hash_changed

        if metrics["updates_applied"] <= 0:
            ineffective_reasons.append("updates_applied<=0")
        if not actor_hash_changed:
            ineffective_reasons.append("actor_param_hash_not_changed")
        if not critic_hash_changed:
            ineffective_reasons.append("critic_param_hash_not_changed")
        learning_effective = len(ineffective_reasons) == 0
    else:
        ineffective_reasons.append("no_train_metrics")

    l1_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l1"])) for ep in episode_outputs)
    l2_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l2"])) for ep in episode_outputs)
    l3_lines = sum(_count_jsonl_lines(Path(ep["logs"]["l3"])) for ep in episode_outputs)
    metrics.update(
        {
            "l1_log_lines": l1_lines,
            "l2_log_lines": l2_lines,
            "l3_log_lines": l3_lines,
        }
    )

    gate_result = gate_eval.evaluate(run_id=run_id, metrics=metrics)

    curriculum_path = artifact_root / "curriculum_state.json"
    gate_path = artifact_root / "gate_result.json"
    summary_path = artifact_root / "pipeline_summary.json"

    checkpoint_layout = _checkpoint_layout(artifact_root)
    saved_checkpoint_paths: list[str] = []
    saved_checkpoint_paths.append(_save_agent_checkpoint(agent, checkpoint_layout["latest"], run_id))
    saved_checkpoint_paths.append(_save_agent_checkpoint(agent, checkpoint_layout["final"], run_id))

    curriculum_path.write_text(
        json.dumps(curriculum.to_artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_gate_report(gate_path, gate_result)

    top_level_target_mode = episode_outputs[-1]["target_mode"] if episode_outputs else target_mode
    top_level_ee_target = episode_outputs[-1]["ee_target"] if episode_outputs else external_ee_target.tolist()
    top_level_ee_target_source = (
        episode_outputs[-1]["ee_target_source"] if episode_outputs else external_ee_target_source
    )

    summary = {
        "run_id": run_id,
        "timestamp_ns": time.time_ns(),
        "episodes": episode_outputs,
        "metrics": metrics,
        "artifacts": {
            "curriculum": str(curriculum_path),
            "gate": str(gate_path),
            "logs_root": str(logs_root),
            "reward_trace": str(reward_trace_path),
            "episode_reward_summary": str(episode_reward_summary_path),
            "runtime_trace": str(runtime_trace_path),
            "checkpoint_latest": str(checkpoint_layout["latest"]),
            "checkpoint_final": str(checkpoint_layout["final"]),
        },
        "policy_mode": policy_mode,
        "runtime_mode": runtime_mode,
        "stage_profile": stage_profile,
        "target_mode": target_mode,
        "resolved_target_mode": top_level_target_mode,
        "gate_overall_decision": gate_result.overall_decision,
        "gate_passed": gate_result.overall_decision == "GO",
        "episode_joint_delta_summary": episode_joint_delta_summary,
        "ee_target": top_level_ee_target,
        "ee_target_source": top_level_ee_target_source,
        "learning_effective": learning_effective,
        "ineffective_reasons": ineffective_reasons,
        "loaded_from_checkpoint": bool(loaded_from_checkpoint),
        "loaded_checkpoint_path": loaded_checkpoint_path,
        "saved_checkpoint_paths": saved_checkpoint_paths,
        "model_persistence_mode": "auto-resume-required",
    }

    if train_metrics:
        summary["train_metrics"] = train_metrics[-20:]

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = "ok"
    exit_code = 0
    if enforce_gates and gate_result.overall_decision != "GO":
        status = "gates_blocked"
        exit_code = 2

    return {
        "summary": str(summary_path),
        "curriculum": str(curriculum_path),
        "gate": str(gate_path),
        "logs_root": str(logs_root),
        "reward_trace": str(reward_trace_path),
        "episode_reward_summary": str(episode_reward_summary_path),
        "runtime_trace": str(runtime_trace_path),
        "status": status,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V5.1 minimal e2e pipeline")
    parser.add_argument("--run-id", default=f"v5_1_e2e_{int(time.time())}")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    parser.add_argument("--artifact-root", default="artifacts/v5_1/e2e")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--policy-mode", choices=["sac_torch"], default="sac_torch")
    parser.add_argument("--runtime-mode", choices=["smoke", "gz"], default="smoke")
    parser.add_argument("--stage-profile", choices=["default", "s0_b"], default="default")
    parser.add_argument("--runtime-joint-names", default="")
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--sac-seed", type=int, default=0)
    parser.add_argument("--ee-pos-success-threshold", type=float, default=0.08)
    parser.add_argument("--ee-ori-success-threshold", type=float, default=0.12)
    parser.add_argument("--external-task-prop", default="tray")
    parser.add_argument("--external-task-src-idx", type=int, default=2)
    parser.add_argument("--external-task-dst-idx", type=int, default=7)
    parser.add_argument("--external-task-waypoint-index", type=int, default=2)
    parser.add_argument("--target-mode", choices=["auto", "external", "near_home"], default="auto")
    parser.add_argument("--near-home-profile", default="s0_bootstrap")
    parser.add_argument("--near-home-pos-offset-min-m", type=float, default=0.02)
    parser.add_argument("--near-home-pos-offset-max-m", type=float, default=0.05)
    parser.add_argument("--near-home-ori-offset-min-deg", type=float, default=5.0)
    parser.add_argument("--near-home-ori-offset-max-deg", type=float, default=10.0)
    args = parser.parse_args()

    joint_names = [x.strip() for x in args.runtime_joint_names.split(",") if x.strip()]

    outputs = run_pipeline_e2e(
        run_id=args.run_id,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        artifact_root=Path(args.artifact_root),
        enforce_gates=args.enforce_gates,
        policy_mode=args.policy_mode,
        sac_seed=args.sac_seed,
        stage_profile=args.stage_profile,
        runtime_mode=args.runtime_mode,
        runtime_joint_names=joint_names,
        trajectory_topic=args.trajectory_topic,
        joint_state_topic=args.joint_state_topic,
        ee_pos_success_threshold=args.ee_pos_success_threshold,
        ee_ori_success_threshold=args.ee_ori_success_threshold,
        external_task_prop=args.external_task_prop,
        external_task_src_idx=args.external_task_src_idx,
        external_task_dst_idx=args.external_task_dst_idx,
        external_task_waypoint_index=args.external_task_waypoint_index,
        target_mode=args.target_mode,
        near_home_profile=args.near_home_profile,
        near_home_pos_offset_min_m=args.near_home_pos_offset_min_m,
        near_home_pos_offset_max_m=args.near_home_pos_offset_max_m,
        near_home_ori_offset_min_deg=args.near_home_ori_offset_min_deg,
        near_home_ori_offset_max_deg=args.near_home_ori_offset_max_deg,
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return int(outputs.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
