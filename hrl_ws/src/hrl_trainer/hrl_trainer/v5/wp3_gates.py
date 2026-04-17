"""WP3 gate runners: WS1 runtime/HIL evidence, WS2 seed sweep stats, WS3 rollback check."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .eval_harness import run_eval_harness


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    # .../RL_brain_trainer/hrl_ws/src/hrl_trainer/hrl_trainer/v5/wp3_gates.py
    return here.parents[5]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class WS1Result:
    gate: str
    passed: bool
    real_path_evidence_ready: bool
    hil_runtime_evidence_ready: bool
    hil_evidence_schema_valid: bool
    hil_evidence_pass: bool
    hil_evidence_checks_all_ok: bool
    hil_evidence_file: str | None
    checked_paths: dict[str, str]
    notes: list[str]
    generated_at_utc: str


@dataclass(frozen=True)
class WS2Result:
    gate: str
    passed: bool
    seeds: list[int]
    episodes: int
    policy: str
    metric: str
    per_seed: list[dict[str, Any]]
    aggregate: dict[str, Any]
    generated_at_utc: str


@dataclass(frozen=True)
class WS3Result:
    gate: str
    passed: bool
    gate_status: str
    rollback_target_policy: str
    strict_policy: bool
    episodes: int
    seeds: list[int]
    aggregate: dict[str, Any]
    per_seed: list[dict[str, Any]]
    checks: dict[str, bool]
    notes: list[str]
    generated_at_utc: str


def _validate_hil_evidence_payload(payload: dict[str, Any]) -> tuple[bool, bool, bool, list[str]]:
    notes: list[str] = []
    required_top = ["timestamp", "runtime_source", "policy", "seed", "checks", "pass", "notes"]
    missing = [k for k in required_top if k not in payload]
    if missing:
        notes.append(f"Missing required top-level fields: {', '.join(missing)}")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        notes.append("checks must be an object containing health/topic/bridge")
        checks = {}

    check_ok_values: list[bool] = []
    for name in ("health", "topic", "bridge"):
        item = checks.get(name)
        if not isinstance(item, dict):
            notes.append(f"checks.{name} must be an object")
            continue
        ok = item.get("ok")
        if not isinstance(ok, bool):
            notes.append(f"checks.{name}.ok must be boolean")
            continue
        check_ok_values.append(ok)

    schema_valid = len(notes) == 0
    pass_field = bool(payload.get("pass", False))
    checks_all_ok = len(check_ok_values) == 3 and all(check_ok_values)
    return schema_valid, pass_field, checks_all_ok, notes


def _find_latest_hil_evidence(hil_dir: Path) -> Path | None:
    candidates = sorted(hil_dir.glob("**/hil_runtime_evidence.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_ws1_runtime_hil_gate(*, repo_root: Path) -> WS1Result:
    checked = {
        "runtime_contract": str(repo_root / "docs/wp3/ws1_runtime_contract.md"),
        "hil_schema": str(repo_root / "docs/wp3/hil_runtime_evidence.schema.json"),
        "hil_profile": str(repo_root / "configs/hil_runtime_profile.yaml"),
        "hil_checklist": str(repo_root / "docs/wp3/hil_env_checklist.md"),
        "real_path_probe": str(repo_root / "artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json"),
        "hil_runtime_evidence_dir": str(repo_root / "artifacts/wp3/hil_dryrun"),
    }

    real_path_evidence_ready = Path(checked["real_path_probe"]).is_file()

    hil_dir = Path(checked["hil_runtime_evidence_dir"])
    hil_latest = _find_latest_hil_evidence(hil_dir)
    hil_runtime_evidence_ready = hil_latest is not None and hil_latest.is_file()

    hil_evidence_schema_valid = False
    hil_evidence_pass = False
    hil_evidence_checks_all_ok = False
    notes: list[str] = []

    if not real_path_evidence_ready:
        notes.append("Missing real path evidence JSON (expected existing v5 benchmark/eval artifact).")

    if not hil_runtime_evidence_ready or hil_latest is None:
        notes.append("No hil_runtime_evidence.json found under artifacts/wp3/hil_dryrun.")
    else:
        try:
            payload = json.loads(hil_latest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("evidence payload root is not object")
            (
                hil_evidence_schema_valid,
                hil_evidence_pass,
                hil_evidence_checks_all_ok,
                validation_notes,
            ) = _validate_hil_evidence_payload(payload)
            notes.extend(validation_notes)

            hil_mode = str(payload.get("mode", "")).strip().lower()
            require_mode = os.environ.get("WP3_HIL_EVIDENCE_MODE", "any").strip().lower()
            if require_mode not in {"any", "non-mock", "real"}:
                require_mode = "any"

            if require_mode == "non-mock" and hil_mode == "mock":
                notes.append("WP3_HIL_EVIDENCE_MODE=non-mock requires evidence.mode != mock.")
                hil_evidence_pass = False
            elif require_mode == "real":
                runtime_source = str(payload.get("runtime_source", "")).strip().lower()
                if hil_mode != "real":
                    notes.append("WP3_HIL_EVIDENCE_MODE=real requires evidence.mode == real.")
                    hil_evidence_pass = False
                if runtime_source.startswith("simulated"):
                    notes.append("WP3_HIL_EVIDENCE_MODE=real rejects simulated runtime_source.")
                    hil_evidence_pass = False
        except Exception as exc:  # pragma: no cover - defensive path
            notes.append(f"Unable to parse HIL evidence JSON: {exc}")

    if hil_runtime_evidence_ready and not hil_evidence_pass:
        notes.append("HIL evidence pass=false.")
    if hil_runtime_evidence_ready and not hil_evidence_checks_all_ok:
        notes.append("HIL checks require health/topic/bridge all ok=true.")

    passed = bool(
        real_path_evidence_ready
        and hil_runtime_evidence_ready
        and hil_evidence_schema_valid
        and hil_evidence_pass
        and hil_evidence_checks_all_ok
    )

    return WS1Result(
        gate="WS1_RUNTIME_HIL",
        passed=passed,
        real_path_evidence_ready=real_path_evidence_ready,
        hil_runtime_evidence_ready=hil_runtime_evidence_ready,
        hil_evidence_schema_valid=hil_evidence_schema_valid,
        hil_evidence_pass=hil_evidence_pass,
        hil_evidence_checks_all_ok=hil_evidence_checks_all_ok,
        hil_evidence_file=str(hil_latest) if hil_latest else None,
        checked_paths=checked,
        notes=notes,
        generated_at_utc=_utc_now(),
    )


def _ci95(mean: float, std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def run_ws2_seed_sweep(*, seeds: list[int], episodes: int, policy: str) -> WS2Result:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    rows: list[dict[str, Any]] = []
    success_rates: list[float] = []
    avg_rewards: list[float] = []

    for seed in seeds:
        summary = run_eval_harness(policy_requested=policy, episodes=episodes, seed=seed, strict_policy=True)
        success_count = int(summary.summary["success_count"])
        fail_count = int(summary.summary["fail_count"])
        total = success_count + fail_count
        success_rate = float(success_count / total) if total > 0 else 0.0
        avg_reward = float(summary.summary["reward"]["average_reward"])
        avg_ep_len = float(summary.summary["reward"]["average_episode_length"])

        row = {
            "seed": int(seed),
            "policy_executed": summary.policy_executed,
            "passed": bool(summary.passed),
            "status_code": int(summary.status_code),
            "success_count": success_count,
            "fail_count": fail_count,
            "success_rate": success_rate,
            "average_reward": avg_reward,
            "average_episode_length": avg_ep_len,
        }
        rows.append(row)
        success_rates.append(success_rate)
        avg_rewards.append(avg_reward)

    n = len(rows)
    sr_mean = float(statistics.mean(success_rates))
    sr_std = float(statistics.stdev(success_rates)) if n > 1 else 0.0
    rw_mean = float(statistics.mean(avg_rewards))
    rw_std = float(statistics.stdev(avg_rewards)) if n > 1 else 0.0

    aggregate = {
        "n": n,
        "success_rate_mean": sr_mean,
        "success_rate_std": sr_std,
        "success_rate_ci95_half_width": _ci95(sr_mean, sr_std, n),
        "average_reward_mean": rw_mean,
        "average_reward_std": rw_std,
        "average_reward_ci95_half_width": _ci95(rw_mean, rw_std, n),
    }

    return WS2Result(
        gate="WS2_STAT_SEED_SWEEP",
        passed=True,
        seeds=[int(s) for s in seeds],
        episodes=int(episodes),
        policy=str(policy),
        metric="success_rate",
        per_seed=rows,
        aggregate=aggregate,
        generated_at_utc=_utc_now(),
    )


def run_ws3_rollback_gate(*, episodes: int, seeds: list[int]) -> WS3Result:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    per_seed: list[dict[str, Any]] = []
    policy_ok = True
    fallback_ok = True
    harness_pass_count = 0
    total_success = 0
    total_fail = 0

    for seed in seeds:
        summary = run_eval_harness(
            policy_requested="rule_l2_v0",
            episodes=episodes,
            seed=int(seed),
            strict_policy=True,
        )
        success_count = int(summary.summary["success_count"])
        fail_count = int(summary.summary["fail_count"])
        total_success += success_count
        total_fail += fail_count
        policy_ok = policy_ok and summary.policy_executed == "rule_l2_v0"
        fallback_ok = fallback_ok and (not summary.fallback_used)
        if summary.passed:
            harness_pass_count += 1

        per_seed.append(
            {
                "seed": int(seed),
                "passed": bool(summary.passed),
                "status_code": int(summary.status_code),
                "success_count": success_count,
                "fail_count": fail_count,
                "summary": summary.to_dict(),
            }
        )

    n = len(seeds)
    all_harness_passed = harness_pass_count == n
    any_harness_passed = harness_pass_count > 0
    aggregate = {
        "seeds_total": n,
        "harness_passed_seeds": harness_pass_count,
        "harness_failed_seeds": n - harness_pass_count,
        "total_success_count": total_success,
        "total_fail_count": total_fail,
    }

    checks = {
        "policy_is_rule_l2_v0": policy_ok,
        "fallback_not_used": fallback_ok,
        "at_least_one_seed_passed": any_harness_passed,
    }

    notes: list[str] = []
    if not all_harness_passed:
        notes.append(
            "Not all rollback seeds passed. Marking conditional_pass when policy/fallback are enforced and at least one seed passes."
        )

    if policy_ok and fallback_ok and all_harness_passed:
        gate_status = "pass"
        passed = True
    elif policy_ok and fallback_ok and any_harness_passed:
        gate_status = "conditional_pass"
        passed = True
    else:
        gate_status = "fail"
        passed = False

    return WS3Result(
        gate="WS3_SAFETY_ROLLBACK",
        passed=passed,
        gate_status=gate_status,
        rollback_target_policy="rule_l2_v0",
        strict_policy=True,
        episodes=int(episodes),
        seeds=[int(s) for s in seeds],
        aggregate=aggregate,
        per_seed=per_seed,
        checks=checks,
        notes=notes,
        generated_at_utc=_utc_now(),
    )


def _parse_seeds(raw: str) -> list[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("--seeds cannot be empty")
    return [int(v) for v in vals]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run WP3 gate checks and emit evidence JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ws1 = sub.add_parser("ws1", help="Runtime/HIL evidence gate")
    ws1.add_argument("--output", required=True)

    ws2 = sub.add_parser("ws2", help="Seed sweep statistics gate")
    ws2.add_argument("--seeds", default="11,13,17")
    ws2.add_argument("--episodes", type=int, default=4)
    ws2.add_argument("--policy", default="rl_l2", choices=["rl_l2", "rule_l2_v0"])
    ws2.add_argument("--output", required=True)
    ws2.add_argument("--per-seed-dir", required=False, default=None)

    ws3 = sub.add_parser("ws3", help="Safety rollback gate")
    ws3.add_argument("--episodes", type=int, default=8)
    ws3.add_argument("--seeds", default="42,43")
    ws3.add_argument("--output", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "ws1":
        repo_root = _repo_root_from_here()
        result = run_ws1_runtime_hil_gate(repo_root=repo_root)
        _write_json(Path(args.output), asdict(result))
        print(json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if result.passed else 2

    if args.cmd == "ws2":
        seeds = _parse_seeds(args.seeds)
        result = run_ws2_seed_sweep(seeds=seeds, episodes=int(args.episodes), policy=str(args.policy))
        payload = asdict(result)

        # optional per-seed evidence fanout
        if args.per_seed_dir:
            per_seed_dir = Path(args.per_seed_dir)
            per_seed_dir.mkdir(parents=True, exist_ok=True)
            for row in payload["per_seed"]:
                seed = int(row["seed"])
                _write_json(per_seed_dir / f"seed_{seed}_summary.json", row)

        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if args.cmd == "ws3":
        seeds = _parse_seeds(args.seeds)
        result = run_ws3_rollback_gate(episodes=int(args.episodes), seeds=seeds)
        _write_json(Path(args.output), asdict(result))
        print(json.dumps(asdict(result), indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if result.passed else 2

    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
