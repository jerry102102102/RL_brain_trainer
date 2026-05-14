import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.phase3a_runtime_node import parse_phase3a_request, run_phase3a_runtime
from hrl_trainer.v5.qwen_l1_client import run_l1_to_rl_input
from hrl_trainer.v5.runtime_model_registry import load_phase3a_model_registry


class TestV5Phase3ARuntime(unittest.TestCase):
    def test_registry_loads_current_approach_finisher_assets(self):
        registry = load_phase3a_model_registry()
        self.assertEqual(registry.pipeline, "APPROACH_FINISHER")
        self.assertIn("workspace_expand_dynscale_stage8_11_big", registry.approach.checkpoint)
        self.assertIn("dock_workspace_handoff_noop", registry.finisher.checkpoint)
        self.assertEqual(registry.ros_topics["skill_command"], "/v5/skill_command")

    def test_parse_qwen_l1_artifact(self):
        result = run_l1_to_rl_input(
            "Move tray1 from shelf_A1 to shelf_B1 while keeping it level.",
            backend="mock_qwen",
        )
        req = parse_phase3a_request(result.to_dict())
        self.assertEqual(req.intent_packet["source_slot"], "shelf_A1")
        self.assertEqual(req.skill_request["pipeline"], "APPROACH -> FINISHER")

    def test_runtime_dry_run_writes_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "request.json"
            output_path = Path(tmp) / "plan.json"
            result = run_l1_to_rl_input(
                "Move tray1 from shelf_A1 to shelf_B1 while keeping it level.",
                backend="mock_qwen",
                output_path=request_path,
            )
            plan = run_phase3a_runtime(request_json=request_path, output_json=output_path)
            saved = json.loads(output_path.read_text())
            self.assertEqual(plan["state_machine"], ["APPROACH", "FINISHER"])
            self.assertEqual(saved["target_slot"], "shelf_B1")
            self.assertFalse(saved["safety_boundary"]["runtime_publishes_joint_trajectory"])
            self.assertIn("path_checks", saved)
            self.assertTrue(saved["path_checks"]["approach"]["checkpoint_exists"])
            self.assertTrue(saved["path_checks"]["approach"]["config_exists"])
            self.assertTrue(saved["path_checks"]["finisher"]["checkpoint_exists"])
            self.assertTrue(saved["path_checks"]["finisher"]["config_exists"])
            self.assertEqual(result.skill_request["pipeline"], "APPROACH -> FINISHER")


if __name__ == "__main__":
    unittest.main()
