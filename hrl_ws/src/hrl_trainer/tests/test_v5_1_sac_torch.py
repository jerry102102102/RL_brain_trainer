from __future__ import annotations

import json
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

    def test_zero_exploration_std_scale_matches_deterministic_action(self) -> None:
        cfg = SACTorchConfig(obs_dim=27, action_dim=7, hidden_dim=64)
        agent = SACTorchAgent(cfg, seed=1)

        obs = np.linspace(-0.25, 0.25, cfg.obs_dim, dtype=np.float32)
        deterministic_action = agent.act(obs, stochastic=False)
        zero_scale_action = agent.act(obs, stochastic=True, exploration_std_scale=0.0)

        self.assertTrue(np.allclose(deterministic_action, zero_scale_action, atol=1e-7))

    def test_lower_exploration_std_scale_reduces_action_deviation(self) -> None:
        cfg = SACTorchConfig(obs_dim=27, action_dim=7, hidden_dim=64)
        agent = SACTorchAgent(cfg, seed=2)

        obs = np.linspace(-0.3, 0.3, cfg.obs_dim, dtype=np.float32)
        deterministic_action = agent.act(obs, stochastic=False)

        torch.manual_seed(123)
        full_scale = np.stack([agent.act(obs, stochastic=True, exploration_std_scale=1.0) for _ in range(64)], axis=0)
        torch.manual_seed(123)
        low_scale = np.stack([agent.act(obs, stochastic=True, exploration_std_scale=0.25) for _ in range(64)], axis=0)

        full_dev = np.linalg.norm(full_scale - deterministic_action[None, :], axis=1).mean()
        low_dev = np.linalg.norm(low_scale - deterministic_action[None, :], axis=1).mean()

        self.assertLess(low_dev, full_dev)

    def test_act_with_diagnostics_exposes_policy_statistics(self) -> None:
        cfg = SACTorchConfig(obs_dim=27, action_dim=7, hidden_dim=64)
        agent = SACTorchAgent(cfg, seed=3)

        obs = np.linspace(-0.1, 0.1, cfg.obs_dim, dtype=np.float32)
        action, diagnostics = agent.act_with_diagnostics(obs, stochastic=True, exploration_std_scale=0.5)

        self.assertEqual(action.shape, (cfg.action_dim,))
        self.assertEqual(len(diagnostics["mu"]), cfg.action_dim)
        self.assertEqual(len(diagnostics["log_std"]), cfg.action_dim)
        self.assertEqual(len(diagnostics["pre_tanh"]), cfg.action_dim)
        self.assertEqual(len(diagnostics["post_tanh"]), cfg.action_dim)
        self.assertEqual(len(diagnostics["final_action"]), cfg.action_dim)
        self.assertEqual(diagnostics["exploration_std_scale"], 0.5)
        self.assertEqual(diagnostics["mu_limit"], cfg.mu_limit)
        self.assertEqual(len(diagnostics["mu_raw"]), cfg.action_dim)
        self.assertGreaterEqual(diagnostics["saturated_dims"], 0)
        self.assertLessEqual(diagnostics["saturated_fraction"], 1.0)

    def test_obs_builder_dim_matches_sac_interface(self) -> None:
        q = np.zeros(7, dtype=np.float32)
        dq = np.zeros(7, dtype=np.float32)
        ee_err = np.zeros(6, dtype=np.float32)
        prev_action = np.zeros(7, dtype=np.float32)
        obs = _obs_from_state(q=q, dq=dq, ee_pose_err=ee_err, prev_action=prev_action)
        self.assertEqual(obs.shape[0], 27)

    def test_train_step_updates_parameters(self) -> None:
        cfg = SACTorchConfig(
            obs_dim=27,
            action_dim=7,
            batch_size=8,
            replay_capacity=256,
            hidden_dim=64,
            actor_update_delay=1,
        )
        agent = SACTorchAgent(cfg, seed=42)

        before_actor = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
        before_q1 = {k: v.detach().clone() for k, v in agent.q1.state_dict().items()}

        for i in range(32):
            obs = np.linspace(-0.2, 0.2, cfg.obs_dim, dtype=np.float32) + i * 1e-3
            raw_action = np.tanh(np.linspace(-0.1, 0.1, cfg.action_dim, dtype=np.float32)) * cfg.action_scale
            exec_action = raw_action * 0.5
            reward = float(1.0 - 0.02 * i)
            next_obs = obs + 0.01
            done = bool(i % 7 == 0)
            agent.remember(
                obs,
                raw_action,
                exec_action,
                reward,
                next_obs,
                done,
                info={
                    "prev_q_des": np.zeros(cfg.action_dim, dtype=np.float32),
                    "next_prev_q_des": np.zeros(cfg.action_dim, dtype=np.float32),
                    "delta_limits": np.full(cfg.action_dim, cfg.action_scale, dtype=np.float32),
                    "delta_norm": 0.5,
                    "raw_norm": 1.0,
                    "exec_norm": 0.5,
                    "clamp_triggered": True,
                    "projection_triggered": False,
                    "rejected": False,
                },
            )

        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("alpha", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("actor_updated", metrics)
        self.assertIn("alpha_updated", metrics)
        self.assertIn("actor_update_delay", metrics)
        self.assertIn("delta_norm_mean", metrics)
        self.assertIn("raw_norm_mean", metrics)
        self.assertIn("exec_norm_mean", metrics)
        self.assertIn("clamp_trigger_rate", metrics)
        self.assertIn("projection_trigger_rate", metrics)
        self.assertIn("reject_rate", metrics)

        actor_changed = any(not torch.allclose(before_actor[k], agent.actor.state_dict()[k]) for k in before_actor)
        q1_changed = any(not torch.allclose(before_q1[k], agent.q1.state_dict()[k]) for k in before_q1)
        self.assertTrue(actor_changed)
        self.assertTrue(q1_changed)

    def test_distill_step_uses_quality_filtered_exec_actions(self) -> None:
        cfg = SACTorchConfig(
            obs_dim=27,
            action_dim=7,
            batch_size=8,
            replay_capacity=256,
            hidden_dim=64,
            action_scale=0.08,
            bc_lambda=0.0,
            distill_lambda=0.1,
            distill_interval=1,
            distill_min_good_count=4,
            distill_candidate_multiplier=4,
            distill_max_delta_norm=1.0,
        )
        agent = SACTorchAgent(cfg, seed=43)

        for i in range(32):
            obs = np.zeros(cfg.obs_dim, dtype=np.float32)
            next_obs = np.zeros(cfg.obs_dim, dtype=np.float32)
            obs[14:17] = np.array([0.09, 0.0, 0.0], dtype=np.float32)
            next_obs[14:17] = np.array([0.02 + 0.0001 * i, 0.0, 0.0], dtype=np.float32)
            raw_action = np.full(cfg.action_dim, 0.02, dtype=np.float32)
            exec_action = raw_action.copy()
            agent.remember(
                obs,
                raw_action,
                exec_action,
                1.0,
                next_obs,
                False,
                info={
                    "prev_q_des": np.zeros(cfg.action_dim, dtype=np.float32),
                    "next_prev_q_des": np.zeros(cfg.action_dim, dtype=np.float32),
                    "delta_limits": np.full(cfg.action_dim, cfg.action_scale, dtype=np.float32),
                    "delta_norm": 0.05,
                    "raw_norm": 0.25,
                    "exec_norm": 0.25,
                    "clamp_triggered": False,
                    "projection_triggered": False,
                    "rejected": False,
                    "success": i < 4,
                    "dwell_count": 1 if i < 8 else 0,
                },
            )

        metrics = agent.train_step()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["distill_enabled"], 1.0)
        self.assertEqual(metrics["distill_triggered"], 1.0)
        self.assertGreater(metrics["distill_update_count"], 0.0)
        self.assertGreaterEqual(metrics["distill_good_count"], 4.0)
        self.assertGreater(metrics["distill_loss"], 0.0)

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
                exploration_std_scale=0.5,
            )

            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["exit_code"], 0)
            self.assertTrue((root / "pipeline_summary.json").exists())
            self.assertTrue((root / "reward_trace.jsonl").exists())
            summary = json.loads((root / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exploration_std_scale"], 0.5)
            self.assertEqual(summary["actor_mu_limit"], 1.5)
            reward_rows = [
                json.loads(x)
                for x in (root / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertIn("policy_debug", reward_rows[0])
            self.assertIn("mu", reward_rows[0]["policy_debug"])
            self.assertIn("mu_raw", reward_rows[0]["policy_debug"])


if __name__ == "__main__":
    unittest.main()
