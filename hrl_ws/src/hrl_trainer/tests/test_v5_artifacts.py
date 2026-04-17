import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.artifacts import build_v5_episode_artifact, write_v5_episode_artifacts
from hrl_trainer.v5.trainer_loop import run_v5_training_episode


class TestV5Artifacts(unittest.TestCase):
    def _sample_episode(self):
        return run_v5_training_episode(
            episode_index=7,
            step_inputs=[
                {
                    "potential_current": 0.0,
                    "potential_next": 0.5,
                    "coverage": 0.1,
                    "subgoal": 0.0,
                    "terminal": False,
                },
                {
                    "potential_current": 0.5,
                    "potential_next": 1.0,
                    "coverage": 0.2,
                    "subgoal": 0.4,
                    "terminal": True,
                    "terminal_success": True,
                },
            ],
        )

    def test_build_artifact_includes_required_fields(self):
        summary = self._sample_episode()
        artifact = build_v5_episode_artifact(
            summary,
            rollout_skill_sequence=["APPROACH", "PLACE", "RETREAT"],
        )

        self.assertEqual(artifact["stage_id"], summary.stage_id)
        self.assertIn("reward_term_totals", artifact)
        self.assertIn("per_step_weighted_terms", artifact)
        self.assertEqual(artifact["rollout_skill_sequence"], ["APPROACH", "PLACE", "RETREAT"])
        self.assertIsNone(artifact["metadata"]["timestamp"])
        self.assertEqual(len(artifact["per_step_weighted_terms"]), 2)

    def test_written_json_and_jsonl_are_stable(self):
        summary = self._sample_episode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            json_a = root / "a.json"
            jsonl_a = root / "a.jsonl"
            json_b = root / "b.json"
            jsonl_b = root / "b.jsonl"

            write_v5_episode_artifacts(
                summary,
                json_path=json_a,
                jsonl_path=jsonl_a,
                rollout_skill_sequence=["APPROACH", "INSERT_SUPPORT", "RETREAT"],
            )
            write_v5_episode_artifacts(
                summary,
                json_path=json_b,
                jsonl_path=jsonl_b,
                rollout_skill_sequence=["APPROACH", "INSERT_SUPPORT", "RETREAT"],
            )

            text_json_a = json_a.read_text(encoding="utf-8")
            text_json_b = json_b.read_text(encoding="utf-8")
            text_jsonl_a = jsonl_a.read_text(encoding="utf-8")
            text_jsonl_b = jsonl_b.read_text(encoding="utf-8")

            self.assertEqual(text_json_a, text_json_b)
            self.assertEqual(text_jsonl_a, text_jsonl_b)

            parsed_json = json.loads(text_json_a)
            parsed_jsonl = [json.loads(line) for line in text_jsonl_a.strip().splitlines()]
            self.assertEqual(parsed_json["stage_id"], summary.stage_id)
            self.assertEqual(parsed_jsonl[0]["record_type"], "episode_summary")
            self.assertEqual(parsed_jsonl[1]["record_type"], "step_weighted_terms")


if __name__ == "__main__":
    unittest.main()
