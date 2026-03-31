from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from .artifact_schema import validate_episode_row, validate_summary
from .train_loop_sac import ToyReachEnv
from .sac_agent import SACAgent


@dataclass(frozen=True)
class GateThresholds:
    success_rate: float
    median_min_goal_error: float
    p90_near_goal_dwell_steps: float
    safety_abort_rate: float
    intervention_rate: float
    p95_episode_length: float | None = None


THRESHOLDS_30 = GateThresholds(0.55, 0.045, 5.0, 0.10, 0.05)
THRESHOLDS_100 = GateThresholds(0.65, 0.035, 6.0, 0.08, 0.03, p95_episode_length=170.0)


def _p(v: list[float], q: float) -> float:
    if not v:
        return 0.0
    return float(np.percentile(np.asarray(v, dtype=np.float32), q))


def evaluate(checkpoint: Path, episodes: int, seed: int, policy_mode: str = "sac", enforce_gates: bool = False, output_dir: Path | None = None) -> tuple[dict[str, Any], int]:
    agent = SACAgent.load(checkpoint)
    env = ToyReachEnv(seed=seed)
    rows: list[dict[str, Any]] = []

    for ep in range(episodes):
        obs = env.reset(seed + ep)
        returns = 0.0
        min_err = float("inf")
        near_dwell = 0
        near_dwell_max = 0
        goal_hits = 0
        safety_count = 0
        intervention_count = 0
        a_l2: list[float] = []
        clamps: list[float] = []
        jerks: list[float] = []
        prev_a = None
        prev2_a = None
        term = "timeout"
        for step in range(env.max_steps):
            action = agent.select_action(obs, deterministic=True) if policy_mode == "sac" else np.zeros(3, dtype=np.float32)
            nxt, info, done, _trunc = env.step(action)
            err = float(info["error"])
            min_err = min(min_err, err)
            near_streak_curr = int(info.get("near_goal_streak_curr", 0))
            near_dwell = near_streak_curr
            near_dwell_max = max(near_dwell_max, near_streak_curr)
            if info.get("goal_hit", False):
                goal_hits += 1
            if info["safety"]:
                safety_count += 1
                term = "safety_abort"
            if info["intervention"]:
                intervention_count += 1
                term = "intervention_abort"
            if info["success"]:
                term = "success"
            a_l2.append(float(np.linalg.norm(action, ord=2)))
            clamps.append(float(info["clamp_ratio"]))
            if prev_a is not None and prev2_a is not None:
                jerks.append(float(np.linalg.norm(action - 2 * prev_a + prev2_a, ord=2)))
            returns += float(1.0 if info["success"] else -err)
            prev2_a = prev_a
            prev_a = action
            obs = nxt
            if done:
                break
        row = {
            "episode_index": ep,
            "seed": seed + ep,
            "return_total": float(returns),
            "terminal_reason": term,
            "success": term == "success",
            "episode_length": int(step + 1),
            "min_goal_error_per_episode": float(min_err),
            "final_goal_error": float(err),
            "near_goal_dwell_steps": int(near_dwell_max),
            "goal_hit_steps": int(goal_hits),
            "safety_violation_count": int(safety_count),
            "intervention_count": int(intervention_count),
            "action_l2_mean": float(np.mean(a_l2) if a_l2 else 0.0),
            "action_l2_p95": float(np.percentile(a_l2, 95) if a_l2 else 0.0),
            "action_clamp_ratio_mean": float(np.mean(clamps) if clamps else 0.0),
            "jerk_l2_mean": float(np.mean(jerks) if jerks else 0.0),
            "timeout_flag": term == "timeout",
            "checkpoint_path": str(checkpoint),
        }
        assert validate_episode_row(row).ok, "invalid episode row"
        rows.append(row)

    success_rate = sum(1 for r in rows if r["success"]) / max(1, len(rows))
    summary = {
        "success_rate": float(success_rate),
        "median_return": float(median([r["return_total"] for r in rows])),
        "p50_episode_length": float(_p([r["episode_length"] for r in rows], 50)),
        "p95_episode_length": float(_p([r["episode_length"] for r in rows], 95)),
        "median_min_goal_error": float(median([r["min_goal_error_per_episode"] for r in rows])),
        "p90_near_goal_dwell_steps": float(_p([r["near_goal_dwell_steps"] for r in rows], 90)),
        "safety_abort_rate": float(sum(1 for r in rows if r["terminal_reason"] == "safety_abort") / len(rows)),
        "intervention_rate": float(sum(r["intervention_count"] for r in rows) / len(rows)),
        "gate_decision": "HOLD",
        "gate_reasons": [],
    }

    if enforce_gates:
        t = THRESHOLDS_30 if episodes <= 30 else THRESHOLDS_100
        reasons: list[str] = []
        if summary["success_rate"] < t.success_rate:
            reasons.append(f"success_rate<{t.success_rate}")
        if summary["median_min_goal_error"] > t.median_min_goal_error:
            reasons.append(f"median_min_goal_error>{t.median_min_goal_error}")
        if summary["p90_near_goal_dwell_steps"] < t.p90_near_goal_dwell_steps:
            reasons.append(f"p90_near_goal_dwell_steps<{t.p90_near_goal_dwell_steps}")
        if summary["safety_abort_rate"] > t.safety_abort_rate:
            reasons.append(f"safety_abort_rate>{t.safety_abort_rate}")
        if summary["intervention_rate"] > t.intervention_rate:
            reasons.append(f"intervention_rate>{t.intervention_rate}")
        if t.p95_episode_length is not None and summary["p95_episode_length"] > t.p95_episode_length:
            reasons.append(f"p95_episode_length>{t.p95_episode_length}")
        summary["gate_reasons"] = reasons
        summary["gate_decision"] = "GO" if not reasons else "HOLD"

    assert validate_summary(summary).ok, "invalid summary"

    status = 0 if (not enforce_gates or summary["gate_decision"] == "GO") else 2

    if output_dir is None:
        output_dir = checkpoint.parent.parent / "e2e" / checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    ep_path = output_dir / "episode_metrics.jsonl"
    sum_path = output_dir / "summary.json"
    gate_path = output_dir / "gate_result.json"
    with ep_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    gate_path.write_text(json.dumps({"episodes": episodes, **summary}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"summary": summary, "episode_metrics": str(ep_path), "summary_path": str(sum_path), "gate_result": str(gate_path)}, status


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate SAC checkpoint and apply gates")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260331)
    p.add_argument("--policy-mode", choices=["rule", "sac"], default="sac")
    p.add_argument("--enforce-gates", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)
    payload, code = evaluate(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        policy_mode=args.policy_mode,
        enforce_gates=args.enforce_gates,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
