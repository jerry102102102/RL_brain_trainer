from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.v5_1.sac_agent import SACAgent, SACConfig


class TestV51SACAgent(unittest.TestCase):
    def test_sac_train_step_updates_and_returns_metrics(self) -> None:
        cfg = SACConfig(obs_dim=12, action_dim=6, batch_size=8)
        agent = SACAgent(cfg, seed=7)

        before_q1 = agent.q1_w.copy()
        before_actor = agent.actor_w.copy()

        for i in range(16):
            obs = np.linspace(-0.1, 0.1, cfg.obs_dim) + i * 1e-3
            act = np.linspace(-0.05, 0.05, cfg.action_dim)
            reward = 1.0 - 0.05 * i
            next_obs = obs + 0.01
            done = i % 5 == 0
            agent.remember(obs, act, reward, next_obs, done)

        metrics = agent.train_step()

        self.assertIsNotNone(metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("alpha", metrics)
        self.assertFalse(np.allclose(before_q1, agent.q1_w))
        self.assertFalse(np.allclose(before_actor, agent.actor_w))


if __name__ == "__main__":
    unittest.main()
