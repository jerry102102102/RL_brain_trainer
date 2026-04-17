import unittest

from hrl_trainer.v5.intent_layer import SlotMap, build_intent_packet
from hrl_trainer.v5.l2_policy import L2_POLICY_RL_L2, L2_POLICY_RULE_V0, build_l2_rollout
from hrl_trainer.v5.rl_action import validate_skill_command_boundary


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


class TestV5RLL2V0(unittest.TestCase):
    def test_rl_l2_rollout_emits_v2_skill_commands_without_l3_fields(self):
        packet = _sample_intent_packet()
        rl_commands = build_l2_rollout(packet, policy_id=L2_POLICY_RL_L2)
        rule_commands = build_l2_rollout(packet, policy_id=L2_POLICY_RULE_V0)

        self.assertTrue(all(cmd.fragility_mode_hint == "CAUTIOUS" for cmd in rl_commands))
        self.assertTrue(all(cmd.speed_profile_id == "NORMAL" for cmd in rl_commands))
        self.assertTrue(all(cmd.speed_profile_id == "SLOW" for cmd in rule_commands))
        self.assertTrue(all(cmd.u_slot_params is not None for cmd in rl_commands))
        self.assertTrue(all(cmd.timing_params is not None for cmd in rl_commands))
        for command in rl_commands:
            validate_skill_command_boundary(command)


if __name__ == "__main__":
    unittest.main()
