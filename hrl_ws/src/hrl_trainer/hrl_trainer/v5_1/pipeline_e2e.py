"""V5.1 end-to-end pipeline (real reward + minimal SAC)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum import CurriculumManager
from .gates import DEFAULT_GATE, GateEvaluator, write_gate_report
from .pipeline_smoke import run_smoke
from .reward import RewardComposer, RewardTraceWriter
from .sac_agent import SACAgent, SACConfig


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


def run_pipeline_e2e(
    run_id: str,
    episodes: int,
    steps_per_episode: int,
    artifact_root: Path,
    enforce_gates: bool = False,
    policy_mode: str = "sac_torch",
    sac_seed: int = 0,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    logs_root = artifact_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    reward_trace_path = artifact_root / "reward_trace.jsonl"
    reward_trace = RewardTraceWriter(reward_trace_path)
    reward_composer = RewardComposer()

    curriculum = CurriculumManager()
    gate_eval = GateEvaluator(DEFAULT_GATE)

    agent: Any | None = None
    if policy_mode == "sac":
        agent = SACAgent(SACConfig(obs_dim=12, action_dim=6), seed=sac_seed)
    elif policy_mode == "sac_torch":
        from .sac_torch import SACTorchAgent, SACTorchConfig

        agent = SACTorchAgent(SACTorchConfig(obs_dim=12, action_dim=6), seed=sac_seed)

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

    for ep in range(episodes_requested):
        stage = curriculum.current_stage
        ep_id = f"{run_id}_ep{ep:03d}_{stage.name}"
        step_count = min(int(steps_per_episode), stage.step_budget)

        target_q = np.array([0.2, -0.15, 0.1, 0.05, 0.0, 0.0], dtype=float)

        def _policy_fn(q: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str]:
            if policy_mode == "rule" or agent is None:
                return (target - q) * 0.5, "rule"
            obs = _obs_from_q_target(q, target)
            return agent.act(obs, stochastic=True), policy_mode

        try:
            logs = run_smoke(run_id=ep_id, steps=step_count, log_root=logs_root, episode=ep, policy_fn=_policy_fn)
        except Exception:
            reset_failures += 1
            break

        expected_log_lines_per_layer += max(1, int(step_count))

        episode_return = 0.0
        ep_intervention = 0
        prev_action = np.zeros(6, dtype=float)
        trace_steps = logs.get("trace_steps", [])

        for idx, step in enumerate(trace_steps):
            action = np.asarray(step["action_raw"], dtype=float)
            prev_error = float(step["goal_error_prev"])
            curr_error = float(step["goal_error_next"])
            intervention_now = step["intervention"] != "none"
            clamp_or_projection = bool(step["saturated"] or step["projection_applied"])

            done = idx == len(trace_steps) - 1
            done_reason = "running"
            if done:
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
                    "done": done,
                    "done_reason": done_reason,
                    "goal_error_prev": prev_error,
                    "goal_error_next": curr_error,
                    "components": terms.to_dict(),
                }
            )

            if intervention_now:
                ep_intervention = 1

            if agent is not None:
                obs = _obs_from_q_target(np.asarray(step["obs_q"], dtype=float), target_q)
                next_q = np.asarray(step["obs_q"], dtype=float) + np.asarray(step["action_clamped"], dtype=float)
                next_obs = _obs_from_q_target(next_q, target_q)
                agent.remember(obs, action, terms.total, next_obs, done)
                train_out = agent.train_step()
                if train_out is not None:
                    train_metrics.append(train_out)

        final_error = float(logs.get("final_goal_error", 1.0))
        ep_success = 1 if final_error < 0.08 else 0

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
            }
        )

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
        },
        "policy_mode": policy_mode,
        "gate_overall_decision": gate_result.overall_decision,
        "gate_passed": gate_result.overall_decision == "GO",
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
    parser.add_argument("--policy-mode", choices=["rule", "sac", "sac_torch"], default="sac_torch")
    parser.add_argument("--sac-seed", type=int, default=0)
    args = parser.parse_args()

    outputs = run_pipeline_e2e(
        run_id=args.run_id,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        artifact_root=Path(args.artifact_root),
        enforce_gates=args.enforce_gates,
        policy_mode=args.policy_mode,
        sac_seed=args.sac_seed,
    )
    print(json.dumps({"run_id": args.run_id, "outputs": outputs}, indent=2, sort_keys=True))
    return int(outputs.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
