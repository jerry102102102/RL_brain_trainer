"""V5.1 end-to-end pipeline (real reward + minimal SAC)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .curriculum import CurriculumManager
from .gates import DEFAULT_GATE, GateEvaluator, write_gate_report
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig
from .pipeline_smoke import run_smoke
from .reward import RewardComposer, RewardTraceWriter
from .runtime_ros2 import RuntimeROS2Adapter


RuntimeFactory = Callable[..., RuntimeROS2Adapter]

_CONTROLLED_ACTION_DIM = 6
_NO_EFFECT_EPS = 1e-6
_NO_EFFECT_STREAK_LIMIT = 3


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _obs_from_q_target(q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    err = target_q - q
    return np.concatenate([q, err], axis=0)


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


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


def _run_episode_gz(
    *,
    ep_id: str,
    ep_index: int,
    step_count: int,
    logs_root: Path,
    runtime: RuntimeROS2Adapter,
    target_q: np.ndarray,
    controlled_indices: list[int],
    policy_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, str]],
    no_effect_epsilon: float = _NO_EFFECT_EPS,
    no_effect_streak_limit: int = _NO_EFFECT_STREAK_LIMIT,
) -> dict[str, Any]:
    ts0 = time.time_ns()
    executor = L3DeterministicExecutor(L3ExecutorConfig(dt=0.1))

    l1_path = logs_root / "l1" / f"{ep_id}.jsonl"
    l2_path = logs_root / "l2" / f"{ep_id}.jsonl"
    l3_path = logs_root / "l3" / f"{ep_id}.jsonl"

    prev_q_des: np.ndarray | None = None
    trace_steps: list[dict[str, Any]] = []
    controlled_idx_np = np.asarray(controlled_indices, dtype=int)
    no_effect_streak = 0

    for step in range(max(1, int(step_count))):
        now_ns = ts0 + step * 100_000_000
        q_before_full = runtime.read_q()
        q_before = q_before_full[controlled_idx_np]
        goal_error_prev = float(np.linalg.norm(target_q - q_before))
        action_raw, policy_name = policy_fn(q_before.copy(), target_q.copy())

        l1_payload = {
            "run_id": ep_id,
            "episode": int(ep_index),
            "step": int(step),
            "ts": int(now_ns),
            "observation": {"q": q_before.tolist(), "target_q": target_q.tolist(), "goal_error_l2": goal_error_prev},
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
        goal_error_next = float(np.linalg.norm(target_q - q_after))

        rt_joint_delta_l2 = float(rt["joint_delta_l2"])
        no_effect = rt_joint_delta_l2 < float(no_effect_epsilon)
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
        }
        _jsonl_append(l3_path, l3_payload)

        intervention = "none"
        if no_effect_streak >= int(no_effect_streak_limit) and goal_error_next >= 0.08:
            intervention = "no_effect"

        trace_steps.append(
            {
                "step": int(step),
                "obs_q": q_before.tolist(),
                "q_after": q_after.tolist(),
                "target_q": target_q.tolist(),
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
    runtime_mode: str = "smoke",
    runtime_factory: RuntimeFactory | None = None,
    runtime_joint_names: list[str] | None = None,
    trajectory_topic: str = "/arm_controller/joint_trajectory",
    joint_state_topic: str = "/joint_states",
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    reward_trace_path = artifact_root / "reward_trace.jsonl"
    reward_trace = RewardTraceWriter(reward_trace_path)
    reward_composer = RewardComposer()

    runtime_trace_path = artifact_root / "runtime_trace.jsonl"
    runtime_trace_path.write_text("", encoding="utf-8")

    curriculum = CurriculumManager()
    gate_eval = GateEvaluator(DEFAULT_GATE)

    if policy_mode != "sac_torch":
        raise ValueError("V5.1 single-path only supports policy_mode=sac_torch")
    if runtime_mode not in {"smoke", "gz"}:
        raise ValueError("runtime_mode must be one of: smoke|gz")

    from .sac_torch import SACTorchAgent, SACTorchConfig

    agent: Any = SACTorchAgent(
        SACTorchConfig(obs_dim=_CONTROLLED_ACTION_DIM * 2, action_dim=_CONTROLLED_ACTION_DIM),
        seed=sac_seed,
    )

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

    try:
        for ep in range(episodes_requested):
            stage = curriculum.current_stage
            ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
            step_count = min(int(steps_per_episode), stage.step_budget)

            target_q = np.array([0.2, -0.15, 0.1, 0.05, 0.0, 0.0], dtype=float)

            def _policy_fn(q: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str]:
                obs = _obs_from_q_target(q, target)
                return agent.act(obs, stochastic=True), policy_mode

            try:
                if runtime_mode == "smoke":
                    logs = run_smoke(
                        run_id=ep_id,
                        steps=step_count,
                        log_root=logs_root,
                        episode=ep,
                        policy_fn=_policy_fn,
                    )
                else:
                    logs = _run_episode_gz(
                        ep_id=ep_id,
                        ep_index=ep,
                        step_count=step_count,
                        logs_root=logs_root,
                        runtime=runtime,
                        target_q=target_q,
                        controlled_indices=runtime_controlled_indices or list(range(_CONTROLLED_ACTION_DIM)),
                        policy_fn=_policy_fn,
                    )
            except Exception:
                reset_failures += 1
                break

            expected_log_lines_per_layer += max(1, int(step_count))

            episode_return = 0.0
            ep_intervention = 0
            prev_action = np.zeros(_CONTROLLED_ACTION_DIM, dtype=float)
            trace_steps = logs.get("trace_steps", [])

            ep_q_before = np.asarray(trace_steps[0]["runtime"]["q_before"], dtype=float) if (trace_steps and runtime_mode == "gz") else None
            ep_q_after = np.asarray(trace_steps[-1]["runtime"]["q_after"], dtype=float) if (trace_steps and runtime_mode == "gz") else None

            for idx, step in enumerate(trace_steps):
                action = np.asarray(step["action_raw"], dtype=float)
                prev_error = float(step["goal_error_prev"])
                curr_error = float(step["goal_error_next"])
                intervention_now = step["intervention"] != "none"
                clamp_or_projection = bool(step["saturated"] or step["projection_applied"])

                done = idx == len(trace_steps) - 1
                done_reason = "running"
                if done:
                    if step["intervention"] == "no_effect":
                        done_reason = "no_effect"
                    else:
                        done_reason = "success" if curr_error < 0.08 else "timeout"

                terms = reward_composer.compute(
                    prev_error=prev_error,
                    curr_error=curr_error,
                    action=action,
                    prev_action=prev_action,
                    intervention=intervention_now,
                    clamp_or_projection=clamp_or_projection,
                    done=done,
                    done_reason=done_reason,
                )
                prev_action = action
                episode_return += terms.total
                reward_totals.append(terms.total)

                reward_trace.append(
                    {
                        "run_id": run_id,
                        "episode": ep,
                        "step": int(step["step"]),
                        "policy_mode": policy_mode,
                        "runtime_mode": runtime_mode,
                        "done": done,
                        "done_reason": done_reason,
                        "goal_error_prev": prev_error,
                        "goal_error_next": curr_error,
                        "components": terms.to_dict(),
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
                            "frame_before_stamp_ns": step["runtime"].get("frame_before_stamp_ns"),
                            "frame_after_stamp_ns": step["runtime"].get("frame_after_stamp_ns"),
                            "goal_error_prev": prev_error,
                            "goal_error_next": curr_error,
                            "no_effect": bool(step.get("no_effect", False)),
                            "no_effect_streak": int(step.get("no_effect_streak", 0)),
                            "timestamp_ns": step["runtime"]["timestamp_ns"],
                        },
                    )

                if intervention_now:
                    ep_intervention = 1

                obs = _obs_from_q_target(np.asarray(step["obs_q"], dtype=float), target_q)
                if runtime_mode == "gz":
                    next_q = np.asarray(step["q_after"], dtype=float)
                else:
                    next_q = np.asarray(step["obs_q"], dtype=float) + np.asarray(step["action_clamped"], dtype=float)
                next_obs = _obs_from_q_target(next_q, target_q)
                agent.remember(obs, action, terms.total, next_obs, done)
                train_out = agent.train_step()
                if train_out is not None:
                    train_metrics.append(train_out)

            final_error = float(logs.get("final_goal_error", 1.0))
            ep_success = 1 if final_error < 0.08 else 0

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

    if train_metrics:
        metrics["train_actor_loss"] = float(np.mean([m["actor_loss"] for m in train_metrics]))
        metrics["train_critic_loss"] = float(np.mean([m["critic_loss"] for m in train_metrics]))
        metrics["train_alpha"] = float(train_metrics[-1]["alpha"])

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

    curriculum_path.write_text(
        json.dumps(curriculum.to_artifact(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_gate_report(gate_path, gate_result)

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
            "runtime_trace": str(runtime_trace_path),
        },
        "policy_mode": policy_mode,
        "runtime_mode": runtime_mode,
        "gate_overall_decision": gate_result.overall_decision,
        "gate_passed": gate_result.overall_decision == "GO",
        "episode_joint_delta_summary": episode_joint_delta_summary,
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
    parser.add_argument("--runtime-joint-names", default="")
    parser.add_argument("--trajectory-topic", default="/arm_controller/joint_trajectory")
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--sac-seed", type=int, default=0)
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
        runtime_mode=args.runtime_mode,
        runtime_joint_names=joint_names,
        trajectory_topic=args.trajectory_topic,
        joint_state_topic=args.joint_state_topic,
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return int(outputs.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
