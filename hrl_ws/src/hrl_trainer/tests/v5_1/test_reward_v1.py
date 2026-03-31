import numpy as np

from hrl_trainer.v5_1.reward_v1 import compute_reward_v1


def test_reward_progress_and_terminal_success_bonus():
    b = compute_reward_v1(
        error_prev=0.1,
        error_curr=0.02,
        yaw_error_curr=0.01,
        action_curr=np.array([0.01, 0.0, 0.0], dtype=np.float32),
        action_prev=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        action_prev2=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        clamp_ratio=0.0,
        safety_violation_event=False,
        intervention_event=False,
        terminal_reason="success",
    )
    assert b.progress > 0
    assert b.goal_hit == 1.0
    assert b.reward_terminal > 0
    assert b.reward_total > b.reward_step


def test_reward_penalizes_safety_and_intervention():
    b = compute_reward_v1(
        error_prev=0.08,
        error_curr=0.09,
        yaw_error_curr=0.2,
        action_curr=np.array([0.1, -0.1, 0.1], dtype=np.float32),
        action_prev=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        action_prev2=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        clamp_ratio=0.5,
        safety_violation_event=True,
        intervention_event=True,
        terminal_reason="safety_abort",
    )
    assert b.safety_violation == 1.0
    assert b.intervention == 1.0
    assert b.reward_terminal < 0
    assert b.reward_total < 0
