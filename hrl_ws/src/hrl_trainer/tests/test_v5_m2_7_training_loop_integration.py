import unittest

from hrl_trainer.v5.artifacts import build_v5_episode_artifact
from hrl_trainer.v5.trainer_loop import run_v5_training_episode, run_v5_training_loop


class TestV5M27TrainingLoopIntegration(unittest.TestCase):
    def test_policy_path_selection_keeps_rule_alias_backward_compatible(self):
        step_inputs = [
            {
                "potential_current": 0.0,
                "potential_next": 1.0,
                "terminal": True,
                "terminal_success": True,
            }
        ]

        summary = run_v5_training_episode(episode_index=5, step_inputs=step_inputs, policy_id="rule_l2")
        self.assertEqual(summary.selected_policy, "rule_l2_v0")
        self.assertEqual(summary.train_counters.train_episode_calls, 0)
        self.assertEqual(summary.train_counters.update_episode_calls, 0)
        self.assertEqual(summary.steps[0].train_counter, 0)
        self.assertEqual(summary.steps[0].update_counter, 0)

    def test_rl_l2_training_path_is_seed_reproducible(self):
        episodes_step_inputs = [
            [
                {
                    "potential_current": 0.0,
                    "potential_next": 0.5,
                    "terminal": False,
                },
                {
                    "potential_current": 0.5,
                    "potential_next": 1.0,
                    "terminal": True,
                    "terminal_success": True,
                },
            ]
            for _ in range(3)
        ]

        run_a = run_v5_training_loop(
            episodes_step_inputs,
            policy_id="rl_l2",
            seed=123,
        )
        run_b = run_v5_training_loop(
            episodes_step_inputs,
            policy_id="rl_l2",
            seed=123,
        )

        self.assertEqual(run_a, run_b)
        self.assertTrue(all(summary.selected_policy == "rl_l2" for summary in run_a))
        self.assertTrue(all(summary.train_counters.train_step_calls > 0 for summary in run_a))
        self.assertTrue(all(summary.train_counters.update_step_calls > 0 for summary in run_a))

    def test_artifact_schema_includes_policy_and_train_update_counters(self):
        summary = run_v5_training_episode(
            episode_index=9,
            step_inputs=[
                {
                    "potential_current": 0.0,
                    "potential_next": 0.5,
                    "terminal": False,
                },
                {
                    "potential_current": 0.5,
                    "potential_next": 1.0,
                    "terminal": True,
                    "terminal_success": True,
                },
            ],
            policy_id="rl_l2",
            seed=7,
        )
        artifact = build_v5_episode_artifact(summary)

        self.assertEqual(artifact["episode_index"], 9)
        self.assertEqual(artifact["stage_id"], summary.stage_id)
        self.assertIn("reward_term_totals", artifact)
        self.assertEqual(artifact["selected_policy"], "rl_l2")
        self.assertIn("train_counters", artifact)
        self.assertIn("train_episode_calls", artifact["train_counters"])
        self.assertIn("update_step_calls", artifact["train_counters"])
        self.assertIn("total_reward", artifact)
        self.assertIn("train_counter", artifact["per_step_weighted_terms"][0])
        self.assertIn("update_counter", artifact["per_step_weighted_terms"][0])


if __name__ == "__main__":
    unittest.main()
