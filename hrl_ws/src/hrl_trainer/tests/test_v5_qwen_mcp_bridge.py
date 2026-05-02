import json
import unittest

from hrl_trainer.v5.qwen_mcp_server import QwenMcpStdioServer
from hrl_trainer.v5.qwen_mcp_tools import QwenMcpBridge, QwenMcpToolError


class TestV5QwenMcpBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = QwenMcpBridge(now_sec=100.0)

    def test_scene_context_lists_slots_and_forbidden_controls(self):
        context = self.bridge.call_tool("get_l1_scene_context", {"include_slot_poses": False})
        self.assertEqual(context["schema_version"], "v5.qwen_mcp.scene_context.v1")
        self.assertIn("shelf_A1", {slot["slot_id"] for slot in context["slots"]})
        self.assertIn("joint_trajectory", context["forbidden_control_outputs"])
        self.assertIn("phase1_approach_to_finisher", context["available_high_level_pipeline"]["name"])

    def test_resolve_intent_packet_from_structured_qwen_proposal(self):
        out = self.bridge.call_tool(
            "resolve_intent_packet",
            {
                "object_id": "tray1",
                "source_slot": "shelf_A1",
                "target_slot": "shelf_B1",
                "constraints": {"speed_cap": "SLOW"},
            },
        )
        packet = out["intent_packet"]
        self.assertEqual(out["status"], "ok")
        self.assertEqual(packet["object_id"], "tray1")
        self.assertEqual(packet["source_slot"], "shelf_A1")
        self.assertEqual(packet["target_slot"], "shelf_B1")
        self.assertEqual(packet["constraints"]["speed_cap"], "SLOW")

    def test_prepare_phase1_skill_request_is_dry_run_only(self):
        resolved = self.bridge.call_tool(
            "resolve_intent_packet",
            {"source_slot": "shelf_A1", "target_slot": "shelf_B1"},
        )
        skill = self.bridge.call_tool(
            "prepare_phase1_skill_request",
            {"intent_packet": resolved["intent_packet"], "dry_run": True},
        )
        self.assertEqual(skill["status"], "accepted_dry_run")
        self.assertEqual(skill["pipeline"], "APPROACH -> FINISHER")
        self.assertIn("approach_checkpoint", skill["phase1_policy_assets"])

        with self.assertRaises(QwenMcpToolError):
            self.bridge.call_tool(
                "prepare_phase1_skill_request",
                {"intent_packet": resolved["intent_packet"], "dry_run": False},
            )

    def test_forbidden_low_level_control_field_is_rejected(self):
        with self.assertRaises(QwenMcpToolError):
            self.bridge.call_tool(
                "resolve_intent_packet",
                {
                    "source_slot": "shelf_A1",
                    "target_slot": "shelf_B1",
                    "joint_trajectory": [],
                },
            )

    def test_stdio_server_lists_and_calls_tools(self):
        server = QwenMcpStdioServer(self.bridge)
        init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "hrl-v5-qwen-mcp")

        listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        self.assertIn("tools", listed["result"])

        called = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "get_l1_scene_context",
                    "arguments": {"include_slot_poses": False},
                },
            }
        )
        text = called["result"]["content"][0]["text"]
        payload = json.loads(text)
        self.assertEqual(payload["schema_version"], "v5.qwen_mcp.scene_context.v1")


if __name__ == "__main__":
    unittest.main()

