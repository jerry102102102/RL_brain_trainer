import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.qwen_l1_client import _extract_json_object, run_l1_to_rl_input


class TestV5QwenL1Client(unittest.TestCase):
    def test_extract_json_from_fenced_model_output(self):
        payload = _extract_json_object(
            """```json
            {"tool":"resolve_intent_packet","arguments":{"source_slot":"shelf_A1","target_slot":"shelf_B1"}}
            ```"""
        )
        self.assertEqual(payload["tool"], "resolve_intent_packet")

    def test_mock_qwen_produces_rl_skill_input_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "l1_to_rl.json"
            result = run_l1_to_rl_input(
                "Move tray1 from shelf_A1 to shelf_B1 while keeping it level.",
                backend="mock_qwen",
                output_path=out_path,
            )
            self.assertTrue(out_path.exists())
            saved = json.loads(out_path.read_text())
            self.assertEqual(saved["tool_call"]["tool"], "resolve_intent_packet")
            self.assertEqual(result.intent_resolution["intent_packet"]["source_slot"], "shelf_A1")
            self.assertEqual(result.intent_resolution["intent_packet"]["target_slot"], "shelf_B1")
            self.assertEqual(result.skill_request["pipeline"], "APPROACH -> FINISHER")
            self.assertEqual(result.skill_request["status"], "accepted_dry_run")


if __name__ == "__main__":
    unittest.main()

