import numpy as np

from hrl_trainer.v5_1.sac_agent import SACAgent, SACConfig


def test_sac_update_smoke_runs_and_returns_metrics():
    cfg = SACConfig(obs_dim=3, action_dim=3, batch_size=32, warmup_steps=0, replay_capacity=2000)
    agent = SACAgent(cfg, seed=7)

    for _ in range(256):
        obs = np.random.uniform(-0.2, 0.2, size=3).astype(np.float32)
        act = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        nxt = obs - 0.2 * act
        rew = float(-np.linalg.norm(nxt))
        done = bool(np.linalg.norm(nxt) < 0.02)
        agent.remember(obs, act, rew, nxt, done, False, info={})

    rows = agent.update()
    assert rows, "expected at least one update row"
    row = rows[0]
    for key in ("critic_loss_1", "critic_loss_2", "actor_loss", "alpha", "alpha_loss", "q_target_mean", "q_target_std", "replay_size"):
        assert key in row
