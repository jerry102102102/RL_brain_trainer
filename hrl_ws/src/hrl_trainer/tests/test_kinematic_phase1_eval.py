from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.envs.arm_kinematic_env import ArmKinematicEnv
from hrl_trainer.kinematic_phase1.eval.eval_approach import run_approach_eval
from hrl_trainer.kinematic_phase1.eval.eval_deterministic import run_deterministic_eval
from hrl_trainer.kinematic_phase1.eval.fixed_eval_suite import build_curriculum_local_eval_suite, build_fixed_eval_suite
from hrl_trainer.kinematic_phase1.eval.metrics import EvalConfig


class TestKinematicPhase1Eval(unittest.TestCase):
    def test_fixed_eval_suite_is_reproducible(self) -> None:
        env = ArmKinematicEnv()
        suite_a = build_fixed_eval_suite(seed=101, n_episodes=3, joint_specs=env.config.joint_specs)
        suite_b = build_fixed_eval_suite(seed=101, n_episodes=3, joint_specs=env.config.joint_specs)
        self.assertEqual([item.goal_q for item in suite_a], [item.goal_q for item in suite_b])
        self.assertEqual([item.initial_q for item in suite_a], [item.initial_q for item in suite_b])

    def test_eval_runner_produces_summary(self) -> None:
        env = ArmKinematicEnv()
        suite = build_fixed_eval_suite(seed=202, n_episodes=2, joint_specs=env.config.joint_specs)

        def env_factory() -> ArmKinematicEnv:
            return ArmKinematicEnv()

        def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
            return np.zeros(7, dtype=np.float32)

        summary = run_deterministic_eval(
            env_factory=env_factory,
            predict_fn=predict_fn,
            suite=suite,
            eval_config=EvalConfig(suite_seed=202, episodes=2),
        )
        self.assertEqual(summary["episode_count"], 2)
        self.assertIn("mean_final_position_error", summary)
        self.assertIn("pre_near_goal_hit_rate", summary)
        self.assertEqual(len(summary["episode_metrics"]), 2)

    def test_approach_eval_produces_position_orientation_breakdown(self) -> None:
        env = ArmKinematicEnv()
        suite = build_fixed_eval_suite(seed=212, n_episodes=2, joint_specs=env.config.joint_specs)

        def env_factory() -> ArmKinematicEnv:
            return ArmKinematicEnv()

        def predict_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
            return np.zeros(7, dtype=np.float32)

        summary = run_approach_eval(
            env_factory=env_factory,
            predict_fn=predict_fn,
            suite=suite,
            eval_config=EvalConfig(suite_seed=212, episodes=2),
        )
        self.assertEqual(summary["episode_count"], 2)
        self.assertIn("pos_only_near_goal_hit_rate", summary)
        self.assertIn("ori_gated_near_goal_hit_rate", summary)
        self.assertIn("failed_due_to_orientation_count", summary)
        self.assertIn("failed_due_to_dwell_count", summary)
        self.assertIn("mean_final_orientation_error", summary)
        self.assertIn("mean_min_orientation_error", summary)

    def test_curriculum_local_eval_suite_uses_fixed_stage_target(self) -> None:
        env = ArmKinematicEnv()
        suite_a = build_curriculum_local_eval_suite(env.config, seed=303, stage_index=0, n_episodes=3)
        suite_b = build_curriculum_local_eval_suite(env.config, seed=303, stage_index=0, n_episodes=3)
        suite_c = build_curriculum_local_eval_suite(env.config, seed=404, stage_index=0, n_episodes=3)
        self.assertEqual(len(suite_a), 3)
        self.assertEqual([item.initial_q for item in suite_a], [item.initial_q for item in suite_b])
        self.assertEqual([item.goal_q for item in suite_a], [item.goal_q for item in suite_b])
        self.assertNotEqual([item.goal_q for item in suite_a], [item.goal_q for item in suite_c])
