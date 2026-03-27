"""Safety watchdog for V5.1 runtime commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Intervention(str, Enum):
    NONE = "none"
    HOLD = "hold"
    STOP = "stop"


@dataclass(frozen=True)
class WatchdogDecision:
    intervention: Intervention
    q_command: np.ndarray | None
    reason: str


class SafetyWatchdog:
    def __init__(self, timeout_s: float, timeout_action: Intervention = Intervention.HOLD) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if timeout_action not in (Intervention.HOLD, Intervention.STOP):
            raise ValueError("timeout_action must be HOLD or STOP")

        self.timeout_s = float(timeout_s)
        self.timeout_action = timeout_action
        self._last_cmd_time_s: float | None = None
        self._hold_q: np.ndarray | None = None

    def observe_command(self, now_s: float, q_current: np.ndarray) -> None:
        self._last_cmd_time_s = float(now_s)
        self._hold_q = np.asarray(q_current, dtype=float).copy()

    def evaluate(self, now_s: float, q_current: np.ndarray) -> WatchdogDecision:
        q_current = np.asarray(q_current, dtype=float)

        if self._last_cmd_time_s is None:
            return WatchdogDecision(Intervention.NONE, None, "no_command_yet")

        elapsed = float(now_s) - float(self._last_cmd_time_s)
        if elapsed <= self.timeout_s:
            return WatchdogDecision(Intervention.NONE, None, "healthy")

        if self.timeout_action == Intervention.HOLD:
            hold_q = self._hold_q if self._hold_q is not None else q_current
            return WatchdogDecision(
                intervention=Intervention.HOLD,
                q_command=np.asarray(hold_q, dtype=float).copy(),
                reason=f"timeout>{self.timeout_s:.3f}s",
            )

        stop_q = np.asarray(q_current, dtype=float).copy()
        return WatchdogDecision(
            intervention=Intervention.STOP,
            q_command=stop_q,
            reason=f"timeout>{self.timeout_s:.3f}s",
        )
