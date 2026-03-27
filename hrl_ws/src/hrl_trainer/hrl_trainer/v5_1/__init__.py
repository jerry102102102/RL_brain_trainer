"""V5.1 namespace: contract-frozen interfaces and deterministic runtime primitives."""

from .contracts import ActionCommand, LayerLogRecord, ObservationFrame, validate_contract
from .l3_executor import L3ExecutorConfig, L3ExecutorResult, L3DeterministicExecutor
from .curriculum import CurriculumManager, CurriculumState, EpisodeRecord, StageSpec
from .gates import DEFAULT_GATE, GateEvaluator, GateResult, GateSpec
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig, L3ExecutorResult
from .safety_watchdog import Intervention, SafetyWatchdog, WatchdogDecision

__all__ = [
    "ActionCommand",
    "LayerLogRecord",
    "ObservationFrame",
    "validate_contract",
    "L3ExecutorConfig",
    "L3ExecutorResult",
    "L3DeterministicExecutor",
    "Intervention",
    "SafetyWatchdog",
    "WatchdogDecision",
    "StageSpec",
    "EpisodeRecord",
    "CurriculumState",
    "CurriculumManager",
    "GateSpec",
    "GateResult",
    "GateEvaluator",
    "DEFAULT_GATE",
]
