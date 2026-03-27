from __future__ import annotations

from hrl_trainer.v5_1.curriculum import CurriculumManager


def test_curriculum_progresses_s0_s1_s2() -> None:
    c = CurriculumManager()

    assert c.current_stage.name == "S0"
    c.record_episode(0.2)
    assert c.current_stage.name == "S0"

    c.record_episode(0.7)
    assert c.current_stage.name == "S1"

    c.record_episode(0.8)
    c.record_episode(0.8)
    assert c.current_stage.name == "S2"


def test_curriculum_terminal_no_overpromote() -> None:
    c = CurriculumManager()
    # Promote to S2
    c.record_episode(0.7)
    c.record_episode(0.7)
    c.record_episode(0.8)
    c.record_episode(0.8)
    assert c.current_stage.name == "S2"

    c.record_episode(1.0)
    c.record_episode(1.0)
    assert c.current_stage.name == "S2"
