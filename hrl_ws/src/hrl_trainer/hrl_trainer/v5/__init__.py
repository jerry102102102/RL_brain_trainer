"""V5 tooling and utilities."""

from .curriculum import CurriculumConfig, CurriculumSelector, CurriculumStageConfig, default_stage_abc_config
from .l2_policy import (
    ACTION_SCHEMA_V1,
    ACTION_SCHEMA_V2,
    L2_POLICY_RL_L2,
    L2_POLICY_RULE_V0,
    L2PolicyAdapter,
    build_l2_rollout,
)
from .rule_l2_v0 import RuleL2V0, RuleL2V0Config

__all__ = [
    "ACTION_SCHEMA_V1",
    "ACTION_SCHEMA_V2",
    "BENCHMARK_SCHEMA",
    "BENCHMARK_VERSION",
    "EVAL_HARNESS_SCHEMA",
    "EVAL_HARNESS_VERSION",
    "CurriculumConfig",
    "CurriculumSelector",
    "CurriculumStageConfig",
    "EvalHarnessSummary",
    "L2_POLICY_RULE_V0",
    "L2_POLICY_RL_L2",
    "L2PolicyAdapter",
    "RuleL2V0BenchmarkSummary",
    "RuleL2V0",
    "RuleL2V0Config",
    "build_v5_episode_artifact",
    "load_rule_l2_v0_benchmark_summary",
    "parse_rule_l2_v0_benchmark_summary",
    "V5EpisodeSummary",
    "V5StepTelemetry",
    "build_l2_rollout",
    "default_stage_abc_config",
    "default_eval_output_path",
    "load_eval_harness_summary",
    "parse_eval_harness_summary",
    "resolve_policy_execution",
    "run_eval_harness",
    "run_rule_l2_v0_benchmark",
    "run_v5_training_episode",
    "run_v5_training_loop",
    "write_eval_harness_summary",
    "write_rule_l2_v0_benchmark_summary",
    "write_v5_episode_artifacts",
    "LearnableL2Policy",
    "SafetyConstrainedL3Executor",
    "run_task1_training",
]


def __getattr__(name: str):
    if name in {"build_v5_episode_artifact", "write_v5_episode_artifacts"}:
        from .artifacts import build_v5_episode_artifact, write_v5_episode_artifacts

        exported = {
            "build_v5_episode_artifact": build_v5_episode_artifact,
            "write_v5_episode_artifacts": write_v5_episode_artifacts,
        }
        return exported[name]
    if name in {
        "BENCHMARK_SCHEMA",
        "BENCHMARK_VERSION",
        "RuleL2V0BenchmarkSummary",
        "run_rule_l2_v0_benchmark",
        "parse_rule_l2_v0_benchmark_summary",
        "load_rule_l2_v0_benchmark_summary",
        "write_rule_l2_v0_benchmark_summary",
    }:
        from .benchmark_rule_l2_v0 import (
            BENCHMARK_SCHEMA,
            BENCHMARK_VERSION,
            RuleL2V0BenchmarkSummary,
            load_rule_l2_v0_benchmark_summary,
            parse_rule_l2_v0_benchmark_summary,
            run_rule_l2_v0_benchmark,
            write_rule_l2_v0_benchmark_summary,
        )

        exported = {
            "BENCHMARK_SCHEMA": BENCHMARK_SCHEMA,
            "BENCHMARK_VERSION": BENCHMARK_VERSION,
            "RuleL2V0BenchmarkSummary": RuleL2V0BenchmarkSummary,
            "run_rule_l2_v0_benchmark": run_rule_l2_v0_benchmark,
            "parse_rule_l2_v0_benchmark_summary": parse_rule_l2_v0_benchmark_summary,
            "load_rule_l2_v0_benchmark_summary": load_rule_l2_v0_benchmark_summary,
            "write_rule_l2_v0_benchmark_summary": write_rule_l2_v0_benchmark_summary,
        }
        return exported[name]
    if name in {
        "EVAL_HARNESS_SCHEMA",
        "EVAL_HARNESS_VERSION",
        "EvalHarnessSummary",
        "default_eval_output_path",
        "load_eval_harness_summary",
        "parse_eval_harness_summary",
        "resolve_policy_execution",
        "run_eval_harness",
        "write_eval_harness_summary",
    }:
        from .eval_harness import (
            EVAL_HARNESS_SCHEMA,
            EVAL_HARNESS_VERSION,
            EvalHarnessSummary,
            default_eval_output_path,
            load_eval_harness_summary,
            parse_eval_harness_summary,
            resolve_policy_execution,
            run_eval_harness,
            write_eval_harness_summary,
        )

        exported = {
            "EVAL_HARNESS_SCHEMA": EVAL_HARNESS_SCHEMA,
            "EVAL_HARNESS_VERSION": EVAL_HARNESS_VERSION,
            "EvalHarnessSummary": EvalHarnessSummary,
            "default_eval_output_path": default_eval_output_path,
            "load_eval_harness_summary": load_eval_harness_summary,
            "parse_eval_harness_summary": parse_eval_harness_summary,
            "resolve_policy_execution": resolve_policy_execution,
            "run_eval_harness": run_eval_harness,
            "write_eval_harness_summary": write_eval_harness_summary,
        }
        return exported[name]
    if name in {"V5EpisodeSummary", "V5StepTelemetry", "run_v5_training_episode", "run_v5_training_loop"}:
        from .trainer_loop import (
            V5EpisodeSummary,
            V5StepTelemetry,
            run_v5_training_episode,
            run_v5_training_loop,
        )

        exported = {
            "V5EpisodeSummary": V5EpisodeSummary,
            "V5StepTelemetry": V5StepTelemetry,
            "run_v5_training_episode": run_v5_training_episode,
            "run_v5_training_loop": run_v5_training_loop,
        }
        return exported[name]
    if name in {"LearnableL2Policy", "SafetyConstrainedL3Executor", "run_task1_training"}:
        from .task1_train import LearnableL2Policy, SafetyConstrainedL3Executor, run_task1_training

        exported = {
            "LearnableL2Policy": LearnableL2Policy,
            "SafetyConstrainedL3Executor": SafetyConstrainedL3Executor,
            "run_task1_training": run_task1_training,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
