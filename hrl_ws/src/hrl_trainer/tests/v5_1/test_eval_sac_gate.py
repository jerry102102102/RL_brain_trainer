from pathlib import Path

import numpy as np

from hrl_trainer.v5_1.eval_sac_gate import evaluate
from hrl_trainer.v5_1.train_loop_sac import ToyReachEnv, run_train


def test_eval_gate_outputs_summary_and_gate_file(tmp_path: Path):
    train_root = tmp_path / "train"
    ckpt = run_train(episodes=3, seed=11, artifact_root=train_root)
    payload, code = evaluate(
        ckpt,
        episodes=5,
        seed=11,
        policy_mode="sac",
        enforce_gates=True,
        output_dir=tmp_path / "e2e",
    )
    assert "summary" in payload
    assert "gate_result" in payload
    assert (tmp_path / "e2e" / "gate_result.json").exists()
    assert code in (0, 2)


def test_dwell_threshold_triggers_success_and_non_dwell_not_success():
    env = ToyReachEnv(seed=0, max_steps=10, success_dwell_steps=3, near_goal_tol=0.05)
    env.reset(episode_seed=1)

    # force near-goal consecutive steps
    env.state[:] = 0.0
    _, info1, done1, _ = env.step(np.zeros(3, dtype=np.float32))
    _, info2, done2, _ = env.step(np.zeros(3, dtype=np.float32))
    _, info3, done3, _ = env.step(np.zeros(3, dtype=np.float32))

    assert info1["near_goal_streak_curr"] == 1
    assert info2["near_goal_streak_curr"] == 2
    assert info3["near_goal_streak_curr"] >= 3
    assert done1 is False and done2 is False
    assert bool(info3["success"]) is True and done3 is True

    env.reset(episode_seed=2)
    env.state[:] = 0.0
    _, info_a, done_a, _ = env.step(np.zeros(3, dtype=np.float32))
    # leave near-goal before reaching threshold
    env.state[:] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    _, info_b, done_b, _ = env.step(np.zeros(3, dtype=np.float32))

    assert info_a["near_goal_streak_curr"] == 1
    assert info_b["near_goal_streak_curr"] == 0
    assert bool(info_b["success"]) is False
    assert done_b is False
