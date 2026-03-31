"""Minimal v5 training/rollout loop integration for curriculum + rewards."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .curriculum import CurriculumSelector, default_stage_abc_config
from .reward_composer import (
    EpisodeRewardBreakdown,
    RewardComposer,
    RewardComposerConfig,
    RewardStepBreakdown,
    RewardTermInput,
    RewardTermWeights,
)


StepInput = RewardTermInput | Mapping[str, object]

POLICY_RULE_L2 = "rule_l2"
POLICY_RULE_L2_V0 = "rule_l2_v0"
POLICY_RL_L2 = "rl_l2"
SUPPORTED_TRAINING_POLICY_IDS = (POLICY_RULE_L2, POLICY_RULE_L2_V0, POLICY_RL_L2)


@dataclass(frozen=True)
class V5StepTelemetry:
    step_index: int
    weighted_terms: dict[str, float]
    raw_terms: dict[str, float]
    total_reward: float
    terminal: bool
    notes: tuple[str, ...]
    train_counter: int = 0
    update_counter: int = 0


@dataclass(frozen=True)
class V5TrainingCounters:
    train_episode_calls: int = 0
    train_step_calls: int = 0
    update_episode_calls: int = 0
    update_step_calls: int = 0


@dataclass(frozen=True)
class V5EpisodeSummary:
    episode_index: int
    stage_id: str
    stage_reward_config: dict[str, float]
    reward_term_totals: dict[str, float]
    total_reward: float
    steps: tuple[V5StepTelemetry, ...]
    selected_policy: str = POLICY_RULE_L2_V0
    train_counters: V5TrainingCounters = field(default_factory=V5TrainingCounters)
    terminal_reason: str | None = None


def _coerce_step_input(step_input: StepInput) -> tuple[RewardTermInput | Mapping[str, object], bool, tuple[str, ...]]:
    if isinstance(step_input, RewardTermInput):
        return step_input, False, ()

    terminal = bool(step_input.get("terminal", False))
    notes_obj = step_input.get("notes", ())
    notes = tuple(str(item) for item in notes_obj) if isinstance(notes_obj, Sequence) and not isinstance(notes_obj, str) else ()

    term_keys = (
        "progress",
        "safety",
        "smoothness",
        "coverage",
        "subgoal",
        "potential_current",
        "potential_next",
        "safety_violation",
        "action_delta",
        "terminal_success",
    )
    terms: dict[str, object] = {}
    for key in term_keys:
        if key in step_input:
            value = step_input[key]
            if key == "terminal_success":
                terms[key] = bool(value)
            else:
                terms[key] = float(value)
    return terms, terminal, notes


def _config_to_mapping(config: RewardComposerConfig | Mapping[str, float] | None) -> dict[str, float]:
    if config is None:
        return {}
    if isinstance(config, RewardComposerConfig):
        return {
            "terminal_success_reward": float(config.terminal_success_reward),
            "terminal_failure_penalty": float(config.terminal_failure_penalty),
            "pbrs_gamma": float(config.pbrs_gamma),
        }
    return {k: float(v) for k, v in config.items()}


def _merge_reward_config(
    base_config: RewardComposerConfig | Mapping[str, float] | None,
    stage_config: Mapping[str, float],
) -> dict[str, float]:
    merged = _config_to_mapping(base_config)
    for key, value in stage_config.items():
        merged[str(key)] = float(value)
    return merged


def _step_to_telemetry(step: RewardStepBreakdown) -> V5StepTelemetry:
    return V5StepTelemetry(
        step_index=step.step_index,
        weighted_terms=dict(step.weighted_terms),
        raw_terms=dict(step.raw_terms),
        total_reward=step.total_reward,
        terminal=step.terminal,
        notes=step.notes,
    )


def normalize_v5_training_policy_id(policy_id: str) -> str:
    normalized = str(policy_id).strip().lower()
    if normalized == POLICY_RULE_L2:
        return POLICY_RULE_L2_V0
    if normalized in SUPPORTED_TRAINING_POLICY_IDS:
        return normalized
    raise ValueError(f"Unsupported training policy_id: {policy_id!r}; expected one of {sorted(SUPPORTED_TRAINING_POLICY_IDS)}")


def _compute_rl_l2_step_counters(*, episode_seed: int, step_index: int, terminal: bool) -> tuple[int, int]:
    # Legacy compatibility only. v5_1 SAC path now owns real train/update counters.
    # Keep zeros here to avoid pseudo/random counters leaking into artifacts.
    _ = (episode_seed, step_index, terminal)
    return 0, 0


def run_v5_training_episode(
    episode_index: int,
    step_inputs: Sequence[StepInput],
    *,
    curriculum_selector: CurriculumSelector | None = None,
    base_reward_config: RewardComposerConfig | Mapping[str, float] | None = None,
    reward_weights: RewardTermWeights | None = None,
    terminal_reason: str | None = None,
    policy_id: str = POLICY_RULE_L2_V0,
    seed: int | None = None,
) -> V5EpisodeSummary:
    selected_policy = normalize_v5_training_policy_id(policy_id)
    selector = curriculum_selector or CurriculumSelector(default_stage_abc_config())
    stage = selector.select_stage(int(episode_index))

    stage_reward_config = _merge_reward_config(base_reward_config, stage.reward_config)
    composer = RewardComposer(weights=reward_weights, config=stage_reward_config)

    step_breakdowns: list[RewardStepBreakdown] = []
    for step_index, payload in enumerate(step_inputs):
        terms, terminal, notes = _coerce_step_input(payload)
        step_breakdowns.append(
            composer.compose_step(
                step_index=step_index,
                terms=terms,
                terminal=terminal,
                notes=notes,
            )
        )

    episode: EpisodeRewardBreakdown = composer.compose_episode(step_breakdowns, terminal_reason=terminal_reason)
    step_telemetry = tuple(_step_to_telemetry(step) for step in episode.steps)

    if selected_policy == POLICY_RL_L2:
        episode_seed = int(seed if seed is not None else 0) + int(episode_index) * 1000003
        rl_steps: list[V5StepTelemetry] = []
        train_step_calls = 0
        update_step_calls = 0
        for step in step_telemetry:
            train_counter, update_counter = _compute_rl_l2_step_counters(
                episode_seed=episode_seed,
                step_index=step.step_index,
                terminal=step.terminal,
            )
            train_step_calls += train_counter
            update_step_calls += update_counter
            rl_steps.append(
                V5StepTelemetry(
                    step_index=step.step_index,
                    weighted_terms=step.weighted_terms,
                    raw_terms=step.raw_terms,
                    total_reward=step.total_reward,
                    terminal=step.terminal,
                    notes=step.notes,
                    train_counter=train_counter,
                    update_counter=update_counter,
                )
            )
        step_telemetry = tuple(rl_steps)
        train_counters = V5TrainingCounters(
            train_episode_calls=1,
            train_step_calls=train_step_calls,
            update_episode_calls=1,
            update_step_calls=update_step_calls,
        )
    else:
        train_counters = V5TrainingCounters()

    return V5EpisodeSummary(
        episode_index=int(episode_index),
        stage_id=stage.stage_id,
        stage_reward_config=stage_reward_config,
        reward_term_totals=dict(episode.term_totals),
        total_reward=episode.total_reward,
        steps=step_telemetry,
        selected_policy=selected_policy,
        train_counters=train_counters,
        terminal_reason=episode.terminal_reason,
    )


def run_v5_training_loop(
    episodes_step_inputs: Sequence[Sequence[StepInput]],
    *,
    episode_start_index: int = 0,
    curriculum_selector: CurriculumSelector | None = None,
    base_reward_config: RewardComposerConfig | Mapping[str, float] | None = None,
    reward_weights: RewardTermWeights | None = None,
    policy_id: str = POLICY_RULE_L2_V0,
    seed: int | None = None,
) -> tuple[V5EpisodeSummary, ...]:
    summaries: list[V5EpisodeSummary] = []
    for offset, step_inputs in enumerate(episodes_step_inputs):
        summaries.append(
            run_v5_training_episode(
                episode_index=episode_start_index + offset,
                step_inputs=step_inputs,
                curriculum_selector=curriculum_selector,
                base_reward_config=base_reward_config,
                reward_weights=reward_weights,
                policy_id=policy_id,
                seed=seed,
            )
        )
    return tuple(summaries)


def _build_smoke_inputs(step_count: int, terminal_success: bool) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    effective_step_count = max(1, step_count)
    for i in range(effective_step_count):
        payload: dict[str, object] = {
            "potential_current": float(i),
            "potential_next": float(i + 1),
            "safety_violation": 0.0,
            "action_delta": 0.1,
            "coverage": 0.05,
            "subgoal": 0.1,
            "terminal": i == effective_step_count - 1,
            "terminal_success": terminal_success,
        }
        if payload["terminal"]:
            payload["notes"] = ("success" if terminal_success else "failure",)
        inputs.append(payload)
    return inputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal v5 curriculum + reward smoke episode.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index used for stage selection.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of synthetic episodes to run.")
    parser.add_argument("--steps", type=int, default=3, help="Number of synthetic steps to run.")
    parser.add_argument("--terminal-success", action="store_true", help="Use terminal success on final step.")
    parser.add_argument("--policy-id", default=POLICY_RULE_L2_V0, choices=sorted(SUPPORTED_TRAINING_POLICY_IDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-output", default=None, help="Optional JSON output for multi-episode smoke summaries.")
    args = parser.parse_args(argv)

    episodes_step_inputs = [_build_smoke_inputs(args.steps, args.terminal_success) for _ in range(max(1, int(args.episodes)))]
    summaries = run_v5_training_loop(
        episodes_step_inputs=episodes_step_inputs,
        episode_start_index=args.episode_index,
        policy_id=args.policy_id,
        seed=args.seed,
    )

    summary = summaries[0]
    print(f"stage_id={summary.stage_id}")
    print(f"selected_policy={summary.selected_policy}")
    print(f"reward_term_totals={summary.reward_term_totals}")
    print(f"train_counters={summary.train_counters}")
    if summary.steps:
        print(f"step_0_weighted_terms={summary.steps[0].weighted_terms}")
        print(f"step_0_train_update={summary.steps[0].train_counter}/{summary.steps[0].update_counter}")

    if args.artifact_output:
        rows = [
            {
                "episode_index": item.episode_index,
                "stage_id": item.stage_id,
                "selected_policy": item.selected_policy,
                "reward_term_totals": item.reward_term_totals,
                "total_reward": item.total_reward,
                "train_counters": {
                    "train_episode_calls": item.train_counters.train_episode_calls,
                    "train_step_calls": item.train_counters.train_step_calls,
                    "update_episode_calls": item.train_counters.update_episode_calls,
                    "update_step_calls": item.train_counters.update_step_calls,
                },
            }
            for item in summaries
        ]
        target = Path(args.artifact_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"artifact_output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
