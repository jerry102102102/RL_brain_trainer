import unittest

from hrl_trainer.v5.curriculum import CurriculumSelector, default_stage_abc_config
from hrl_trainer.v5.trainer_loop import run_v5_training_episode, run_v5_training_loop


class TestV5M23IntegrationLoop(unittest.TestCase):
    def test_stage_selection_changes_with_episode_index(self):
        selector = CurriculumSelector(default_stage_abc_config())
        common_steps = [
            {
                "potential_current": 0.0,
                "potential_next": 0.5,
                "terminal": True,
                "terminal_success": True,
            }
        ]

        summary_a = run_v5_training_episode(episode_index=0, step_inputs=common_steps, curriculum_selector=selector)
        summary_b = run_v5_training_episode(episode_index=1200, step_inputs=common_steps, curriculum_selector=selector)
        summary_c = run_v5_training_episode(episode_index=4000, step_inputs=common_steps, curriculum_selector=selector)

        self.assertEqual(summary_a.stage_id, "A")
        self.assertEqual(summary_b.stage_id, "B")
        self.assertEqual(summary_c.stage_id, "C")

    def test_reward_uses_stage_terminal_config(self):
        selector = CurriculumSelector(default_stage_abc_config())
        steps = [
            {
                "potential_current": 0.0,
                "potential_next": 0.0,
                "terminal": True,
                "terminal_success": True,
            }
        ]

        summary_a = run_v5_training_episode(episode_index=0, step_inputs=steps, curriculum_selector=selector)
        summary_b = run_v5_training_episode(episode_index=1200, step_inputs=steps, curriculum_selector=selector)
        summary_c = run_v5_training_episode(episode_index=4000, step_inputs=steps, curriculum_selector=selector)

        self.assertAlmostEqual(summary_a.reward_term_totals["sparse_terminal"], 1.0, places=6)
        self.assertAlmostEqual(summary_b.reward_term_totals["sparse_terminal"], 2.0, places=6)
        self.assertAlmostEqual(summary_c.reward_term_totals["sparse_terminal"], 3.0, places=6)

    def test_telemetry_contains_stage_and_reward_breakdown_keys(self):
        loop_summaries = run_v5_training_loop(
            episodes_step_inputs=[
                [
                    {
                        "progress": 0.1,
                        "safety": -0.2,
                        "smoothness": -0.05,
                        "terminal": True,
                        "terminal_success": False,
                    }
                ]
            ],
            episode_start_index=1000,
        )

        self.assertEqual(len(loop_summaries), 1)
        summary = loop_summaries[0]

        self.assertEqual(summary.stage_id, "B")
        self.assertTrue(summary.steps)

        expected_keys = {
            "sparse_terminal",
            "pbrs_delta",
            "safety_penalty",
            "smoothness_penalty",
            "coverage",
            "subgoal",
        }
        self.assertTrue(expected_keys.issubset(summary.reward_term_totals.keys()))
        self.assertTrue(expected_keys.issubset(summary.steps[0].weighted_terms.keys()))


if __name__ == "__main__":
    unittest.main()
