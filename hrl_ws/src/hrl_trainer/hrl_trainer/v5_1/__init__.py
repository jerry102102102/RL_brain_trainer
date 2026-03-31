"""V5.1 namespace: contract-frozen interfaces and deterministic runtime primitives."""

from .contracts import ActionCommand, LayerLogRecord, ObservationFrame, validate_contract
from .l3_executor import L3ExecutorConfig, L3ExecutorResult, L3DeterministicExecutor
from .curriculum import CurriculumManager, CurriculumState, EpisodeRecord, StageSpec
from .gates import DEFAULT_GATE, GateEvaluator, GateResult, GateSpec
from .l3_executor import L3DeterministicExecutor, L3ExecutorConfig, L3ExecutorResult
from .safety_watchdog import Intervention, SafetyWatchdog, WatchdogDecision
from .reward import RewardComposer, RewardConfig, RewardTerms
from .reward_v1 import RewardV1Breakdown, RewardV1Config, compute_reward_v1
from .replay_buffer import ReplayBuffer
from .sac_agent import SACAgent, SACConfig

try:
    from .sac_torch import SACTorchAgent, SACTorchConfig, TorchReplayBuffer
except ModuleNotFoundError:  # pragma: no cover - optional dependency for torch path
    SACTorchAgent = None
    SACTorchConfig = None
    TorchReplayBuffer = None

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
    "RewardComposer",
    "RewardConfig",
    "RewardTerms",
    "RewardV1Config",
    "RewardV1Breakdown",
    "compute_reward_v1",
    "ReplayBuffer",
    "SACConfig",
    "SACAgent",
]

if SACTorchAgent is not None:
    __all__.extend(["SACTorchAgent", "SACTorchConfig", "TorchReplayBuffer"])
