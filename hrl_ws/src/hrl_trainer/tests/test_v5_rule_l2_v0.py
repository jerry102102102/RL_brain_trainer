import unittest

from hrl_trainer.v5.intent_layer import SlotMap, build_intent_packet
from hrl_trainer.v5.l2_policy import ACTION_SCHEMA_V1, L2_POLICY_RULE_V0, L2PolicyAdapter, build_l2_rollout
from hrl_trainer.v5.rl_action import validate_skill_command_boundary
from hrl_trainer.v5.rule_l2_v0 import RuleL2V0


def _slotmap_payload():
    return {
        "slots": [
            {
                "slot_id": "shelf_A1",
                "region_world": {"center_xyz": [0.9, -1.16, 1.22], "size_xyz": [0.18, 0.18, 0.06], "yaw": 0.0},
                "approach_pose_candidates": [{"xyz": [0.86, -1.10, 1.32], "rpy": [3.14, 0.0, 0.0]}],
                "place_pose_candidates": [{"xyz": [0.90, -1.16, 1.22], "rpy": [3.14, 0.0, 0.0]}],
                "allowed_objects": ["tray1"],
                "priority": 1,
            },
            {
                "slot_id": "shelf_B1",
                "region_world": {"center_xyz": [-0.92, -1.16, 1.22], "size_xyz": [0.18, 0.18, 0.06], "yaw": 0.0},
                "approach_pose_candidates": [{"xyz": [-0.86, -1.10, 1.32], "rpy": [3.14, 0.0, 3.14]}],
                "place_pose_candidates": [{"xyz": [-0.92, -1.16, 1.22], "rpy": [3.14, 0.0, 3.14]}],
                "allowed_objects": ["tray1"],
                "priority": 1,
            },
        ]
    }


def _sample_intent_packet():
    slot_map = SlotMap.from_dict(_slotmap_payload())
    return build_intent_packet(
        "MOVE_PLATE(shelf_A1, shelf_B1)",
        slot_map,
        [{"object_id": "tray1", "confidence": 0.9, "stamp_sec": 10.0}],
        now_sec=10.1,
    )


class TestV5RuleL2V0(unittest.TestCase):
    def test_rollout_is_deterministic_and_boundary_safe(self):
        packet = _sample_intent_packet()
        policy = RuleL2V0()
        rollout_a = policy.rollout(packet)
        rollout_b = policy.rollout(packet)

        self.assertEqual([cmd.skill_mode for cmd in rollout_a], ["APPROACH", "INSERT_SUPPORT", "LIFT_CARRY", "PLACE", "RETREAT"])
        self.assertEqual([cmd.skill_mode for cmd in rollout_a], [cmd.skill_mode for cmd in rollout_b])

        for command in rollout_a:
            validate_skill_command_boundary(command)

    def test_terminal_skill_boundary(self):
        with self.assertRaises(ValueError):
            L2PolicyAdapter(policy_id=L2_POLICY_RULE_V0, terminal_skill="PLACE")

    def test_policy_selector_for_runtime_rollout(self):
        packet = _sample_intent_packet()
        commands = build_l2_rollout(packet, policy_id=L2_POLICY_RULE_V0, terminal_skill="WITHDRAW")
        self.assertEqual([cmd.skill_mode for cmd in commands], ["APPROACH", "INSERT_SUPPORT", "LIFT_CARRY", "PLACE", "WITHDRAW"])
        for command in commands:
            self.assertIsNotNone(command.u_slot_params)
            self.assertIsNotNone(command.timing_params)
            self.assertEqual(command.gripper_cmd, "HOLD")

    def test_explicit_v1_action_schema_keeps_compatibility_path(self):
        packet = _sample_intent_packet()
        commands = build_l2_rollout(packet, policy_id=L2_POLICY_RULE_V0, action_schema=ACTION_SCHEMA_V1)

        self.assertEqual([cmd.skill_mode for cmd in commands], ["APPROACH", "INSERT_SUPPORT", "LIFT_CARRY", "PLACE", "RETREAT"])
        self.assertTrue(any(cmd.gripper_cmd in {"OPEN", "CLOSE"} for cmd in commands))
        for command in commands:
            self.assertIsNone(command.u_slot_params)
            self.assertIsNone(command.timing_params)
            validate_skill_command_boundary(command)

    def test_action_schema_selector_rejects_unsupported_value(self):
        packet = _sample_intent_packet()
        with self.assertRaisesRegex(ValueError, "Unsupported action_schema"):
            build_l2_rollout(packet, policy_id=L2_POLICY_RULE_V0, action_schema="v3")


if __name__ == "__main__":
    unittest.main()
