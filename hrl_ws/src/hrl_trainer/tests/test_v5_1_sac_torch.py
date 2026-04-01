from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import importlib.util

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

from hrl_trainer.v5_1.pipeline_e2e import _obs_from_state, run_pipeline_e2e

if TORCH_AVAILABLE:
    from hrl_trainer.v5_1.sac_torch import SACTorchAgent, SACTorchConfig


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestV51SACTorch(unittest.TestCase):
    def test_forward_action_shape(self) -> None:
        cfg = SACTorchConfig(obs_dim=27, action_dim=7, hidden_dim=64)
        agent = SACTorchAgent(cfg, seed=0)

        obs = np.linspace(-0.5, 0.5, cfg.obs_dim, dtype=np.float32)
        action = agent.act(obs, stochastic=True)

        self.assertEqual(action.shape, (cfg.action_dim,))

    def test_obs_builder_dim_matches_sac_interface(self) -> None:
        q = np.zeros(7, dtype=np.float32)
        dq = np.zeros(7, dtype=np.float32)
        ee_err = np.zeros(6, dtype=np.float32)
        prev_action = np.zeros(7, dtype=np.float32)
        obs = _obs_from_state(q=q, dq=dq, ee_pose_err=ee_err, prev_action=prev_action)
        self.assertEqual(obs.shape[0], 27)

    def test_train_step_updates_parameters(self) -> None:
        cfg = SACTorchConfig(obs_dim=27, action_dim=7, batch_size=8, replay_capacity=256, hidden_dim=64)
        agent = SACTorchAgent(cfg, seed=42)

        before_actor = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
        before_q1 = {k: v.detach().clone() for k, v in agent.q1.state_dict().items()}

        for i in range(32):
            obs = np.linspace(-0.2, 0.2, cfg.obs_dim, dtype=np.float32) + i * 1e-3
            action = np.tanh(np.linspace(-0.1, 0.1, cfg.action_dim, dtype=np.float32)) * cfg.action_scale
            reward = float(1.0 - 0.02 * i)
            next_obs = obs + 0.01
            done = bool(i % 7 == 0)
            agent.remember(obs, action, reward, next_obs, done)

        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("alpha", metrics)
        self.assertIn("entropy", metrics)

        actor_changed = any(not torch.allclose(before_actor[k], agent.actor.state_dict()[k]) for k in before_actor)
        q1_changed = any(not torch.allclose(before_q1[k], agent.q1.state_dict()[k]) for k in before_q1)
        self.assertTrue(actor_changed)
        self.assertTrue(q1_changed)

    def test_pipeline_e2e_runs_with_sac_torch_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            out = run_pipeline_e2e(
                run_id="test_e2e_sac_torch",
                episodes=3,
                steps_per_episode=12,
                artifact_root=root,
                policy_mode="sac_torch",
                sac_seed=7,
            )

            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["exit_code"], 0)
            self.assertTrue((root / "pipeline_summary.json").exists())
            self.assertTrue((root / "reward_trace.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
