import unittest

from hrl_trainer.v5.curriculum import CurriculumSelector, default_stage_abc_config
from hrl_trainer.v5.reward_composer import RewardComposer, RewardComposerConfig, RewardTermInput, RewardTermWeights


class TestRewardComposerV2(unittest.TestCase):
    def test_v2_component_arithmetic(self):
        composer = RewardComposer(
            weights=RewardTermWeights(progress=2.0, safety=3.0, smoothness=4.0, coverage=0.0, subgoal=0.0),
            config=RewardComposerConfig(
                terminal_success_reward=5.0,
                terminal_failure_penalty=-5.0,
                pbrs_gamma=0.9,
            ),
        )

        step = composer.compose_step(
            7,
            RewardTermInput(
                potential_current=1.0,
                potential_next=3.0,
                safety_violation=0.5,
                action_delta=0.25,
                terminal_success=True,
            ),
            terminal=True,
        )

        # PBRS delta = 0.9*3 - 1 = 1.7 => *2 = 3.4
        # safety penalty = -0.5 => *3 = -1.5
        # smoothness penalty = -0.25 => *4 = -1.0
        # terminal sparse = +5.0
        expected = 5.0 + 3.4 - 1.5 - 1.0
        self.assertAlmostEqual(step.total_reward, expected, places=6)
        self.assertAlmostEqual(step.weighted_terms["pbrs_delta"], 3.4, places=6)
        self.assertAlmostEqual(step.weighted_terms["safety_penalty"], -1.5, places=6)
        self.assertAlmostEqual(step.weighted_terms["smoothness_penalty"], -1.0, places=6)
        self.assertAlmostEqual(step.weighted_terms["sparse_terminal"], 5.0, places=6)


class TestCurriculumStageSelector(unittest.TestCase):
    def test_default_stage_boundaries(self):
        selector = CurriculumSelector(default_stage_abc_config())
        self.assertEqual(selector.select_stage(0).stage_id, "A")
        self.assertEqual(selector.select_stage(1200).stage_id, "B")
        self.assertEqual(selector.select_stage(99999).stage_id, "C")

    def test_selector_hook_override(self):
        selector = CurriculumSelector(
            default_stage_abc_config(),
            selector_hook=lambda episode_index, cfg: "C" if episode_index < 10 else None,
        )
        self.assertEqual(selector.select_stage(1).stage_id, "C")
        self.assertEqual(selector.select_stage(1000).stage_id, "B")


if __name__ == "__main__":
    unittest.main()
