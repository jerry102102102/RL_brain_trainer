"""Unified v5 evaluation harness wrapper with policy selection and fallback."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .benchmark_rl_l2_v0 import run_rl_l2_v0_benchmark
from .benchmark_rule_l2_v0 import run_rule_l2_v0_benchmark

EVAL_HARNESS_SCHEMA = "v5_eval_harness"
EVAL_HARNESS_VERSION = "1.0"
POLICY_RULE_L2_V0 = "rule_l2_v0"
POLICY_RL_L2 = "rl_l2"
SUPPORTED_POLICY_IDS = frozenset({POLICY_RULE_L2_V0, POLICY_RL_L2})
BenchmarkRunner = Callable[..., Any]


@dataclass(frozen=True)
class EvalHarnessSummary:
    schema: str
    version: str
    policy_requested: str
    policy_executed: str
    fallback_used: bool
    seed: int
    episodes: int
    summary: dict[str, Any]
    status_code: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "policy_requested": self.policy_requested,
            "policy_executed": self.policy_executed,
            "fallback_used": bool(self.fallback_used),
            "seed": int(self.seed),
            "episodes": int(self.episodes),
            "summary": dict(self.summary),
            "status_code": int(self.status_code),
            "passed": bool(self.passed),
        }


def _detect_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "artifacts").exists() and (candidate / "hrl_ws").exists():
            return candidate
    return start


def default_eval_output_path(*, policy_requested: str, seed: int, episodes: int, cwd: Path | None = None) -> Path:
    base = _detect_repo_root((cwd or Path.cwd()).resolve())
    return base / "artifacts" / "reports" / "v5" / f"v5_eval_{policy_requested}_seed{seed}_ep{episodes}.json"


def _default_policy_runners() -> dict[str, BenchmarkRunner]:
    return {
        POLICY_RULE_L2_V0: run_rule_l2_v0_benchmark,
        POLICY_RL_L2: run_rl_l2_v0_benchmark,
    }


def require_policy_implementation(
    policy_id: str,
    *,
    policy_runners: Mapping[str, BenchmarkRunner] | None = None,
) -> None:
    if policy_id not in (policy_runners or _default_policy_runners()):
        raise ValueError(f"Policy {policy_id!r} is not implemented")


def resolve_policy_execution(
    policy_requested: str,
    *,
    strict_policy: bool = False,
    policy_runners: Mapping[str, BenchmarkRunner] | None = None,
) -> tuple[str, bool]:
    runners = policy_runners or _default_policy_runners()
    policy_id = str(policy_requested)
    if policy_id not in SUPPORTED_POLICY_IDS:
        raise ValueError(f"Unsupported policy id {policy_id!r}; expected one of {sorted(SUPPORTED_POLICY_IDS)}")
    if policy_id in runners:
        return policy_id, False
    if strict_policy:
        raise ValueError(f"Policy {policy_id!r} is not implemented; strict mode forbids fallback")
    require_policy_implementation(POLICY_RULE_L2_V0, policy_runners=runners)
    return POLICY_RULE_L2_V0, policy_id != POLICY_RULE_L2_V0


def run_eval_harness(
    *,
    policy_requested: str = POLICY_RULE_L2_V0,
    episodes: int = 8,
    seed: int = 42,
    strict_policy: bool = False,
    policy_mode: str = "rule",
    enforce_gates: bool = False,
    checkpoint: str | None = None,
) -> EvalHarnessSummary:
    if policy_mode == "sac" or enforce_gates or checkpoint:
        raise ValueError(
            "Legacy v5.1 SAC gate evaluation has been removed from the maintained V5.1 mainline. "
            "Use the active v5_1 pipeline/eval flow instead of eval_harness for SAC-based evaluation."
        )

    policy_executed, fallback_used = resolve_policy_execution(policy_requested, strict_policy=strict_policy)
    benchmark_runner = _default_policy_runners()[policy_executed]

    benchmark_summary = benchmark_runner(episodes=episodes, seed=seed)
    pass_condition = benchmark_summary.success_count >= benchmark_summary.fail_count
    status_code = 0 if pass_condition else 1
    summary_payload = {
        "stage": {
            "name": "l2_policy_benchmark",
            "benchmark_schema": benchmark_summary.schema,
            "benchmark_version": benchmark_summary.version,
        },
        "reward": {
            "average_reward": float(benchmark_summary.average_reward),
            "average_episode_length": float(benchmark_summary.average_episode_length),
        },
        "success_count": int(benchmark_summary.success_count),
        "fail_count": int(benchmark_summary.fail_count),
    }
    return EvalHarnessSummary(
        schema=EVAL_HARNESS_SCHEMA,
        version=EVAL_HARNESS_VERSION,
        policy_requested=str(policy_requested),
        policy_executed=policy_executed,
        fallback_used=bool(fallback_used),
        seed=int(seed),
        episodes=int(episodes),
        summary=summary_payload,
        status_code=int(status_code),
        passed=bool(pass_condition),
    )


def _validate_summary_payload(summary: Mapping[str, Any]) -> None:
    legacy_required = {"stage", "reward", "success_count", "fail_count"}
    gate_required = {"success_rate", "median_min_goal_error", "gate_decision", "gate_reasons"}
    has_legacy = legacy_required.issubset(set(summary.keys()))
    has_gate = gate_required.issubset(set(summary.keys()))
    if not has_legacy and not has_gate:
        raise ValueError("Eval harness summary must be legacy benchmark payload or gate payload")
    if has_legacy:
        if not isinstance(summary["stage"], Mapping):
            raise ValueError("summary.stage must be a JSON object")
        if not isinstance(summary["reward"], Mapping):
            raise ValueError("summary.reward must be a JSON object")


def parse_eval_harness_summary(payload: Mapping[str, Any]) -> EvalHarnessSummary:
    required = {
        "schema",
        "version",
        "policy_requested",
        "policy_executed",
        "fallback_used",
        "seed",
        "episodes",
        "summary",
        "status_code",
        "passed",
    }
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Eval harness payload missing required fields: {missing}")
    if not isinstance(payload["summary"], Mapping):
        raise ValueError("summary must be a JSON object")
    summary_payload = dict(payload["summary"])
    _validate_summary_payload(summary_payload)

    parsed = EvalHarnessSummary(
        schema=str(payload["schema"]),
        version=str(payload["version"]),
        policy_requested=str(payload["policy_requested"]),
        policy_executed=str(payload["policy_executed"]),
        fallback_used=bool(payload["fallback_used"]),
        seed=int(payload["seed"]),
        episodes=int(payload["episodes"]),
        summary=summary_payload,
        status_code=int(payload["status_code"]),
        passed=bool(payload["passed"]),
    )
    if parsed.episodes <= 0:
        raise ValueError("episodes must be > 0")
    return parsed


def load_eval_harness_summary(path: str | Path) -> EvalHarnessSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Eval harness payload must be a JSON object")
    return parse_eval_harness_summary(payload)


def write_eval_harness_summary(summary: EvalHarnessSummary, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    target.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v5 eval harness with baseline fallback support")
    parser.add_argument("--policy", default=POLICY_RULE_L2_V0, choices=sorted(SUPPORTED_POLICY_IDS))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-policy", action="store_true")
    parser.add_argument("--policy-mode", choices=["rule", "sac"], default="rule")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    summary = run_eval_harness(
        policy_requested=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        strict_policy=args.strict_policy,
        policy_mode=args.policy_mode,
        enforce_gates=args.enforce_gates,
        checkpoint=args.checkpoint,
    )
    output_path = Path(args.output) if args.output else default_eval_output_path(
        policy_requested=args.policy,
        seed=args.seed,
        episodes=args.episodes,
    )
    write_eval_harness_summary(summary, output_path)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
