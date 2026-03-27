from __future__ import annotations

import numpy as np

from hrl_trainer.v5_1.safety_watchdog import Intervention, SafetyWatchdog


def test_watchdog_timeout_hold() -> None:
    wd = SafetyWatchdog(timeout_s=0.3, timeout_action=Intervention.HOLD)
    q0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    wd.observe_command(now_s=1.0, q_current=q0)

    healthy = wd.evaluate(now_s=1.2, q_current=q0)
    assert healthy.intervention == Intervention.NONE

    timeout = wd.evaluate(now_s=1.31, q_current=np.zeros(6, dtype=float))
    assert timeout.intervention == Intervention.HOLD
    assert np.allclose(timeout.q_command, q0)


def test_watchdog_timeout_stop() -> None:
    wd = SafetyWatchdog(timeout_s=0.2, timeout_action=Intervention.STOP)
    q = np.array([0.2, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=float)
    wd.observe_command(now_s=0.0, q_current=q)

    timeout = wd.evaluate(now_s=0.25, q_current=q)
    assert timeout.intervention == Intervention.STOP
    assert np.allclose(timeout.q_command, q)
