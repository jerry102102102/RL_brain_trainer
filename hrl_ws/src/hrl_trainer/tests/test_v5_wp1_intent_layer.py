import unittest

from hrl_trainer.v5.intent_layer import (
    IntentFailureCode,
    IntentResolutionError,
    IntentValidationError,
    SlotMap,
    build_intent_packet,
    parse_move_plate,
    validate_intent_packet,
)
from hrl_trainer.v5.perception_adapter import PerceptionAdapter, PerceptionAdapterConfig


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


class TestV5Wp1IntentLayer(unittest.TestCase):
    def test_parse_move_plate(self):
        src, dst = parse_move_plate("MOVE_PLATE(shelf_A1, shelf_B1)")
        self.assertEqual(src, "shelf_A1")
        self.assertEqual(dst, "shelf_B1")

    def test_intent_packet_validator_accepts_valid_packet(self):
        slot_map = SlotMap.from_dict(_slotmap_payload())
        packet = build_intent_packet(
            "MOVE_PLATE(shelf_A1, shelf_B1)",
            slot_map,
            [{"object_id": "tray1", "confidence": 0.9, "stamp_sec": 10.0}],
            now_sec=10.1,
        )
        validate_intent_packet(packet)
        self.assertEqual(packet.object_id, "tray1")

    def test_intent_packet_validator_blocks_l2_l3_fields(self):
        bad_payload = {
            "object_id": "tray1",
            "source_slot": "shelf_A1",
            "target_slot": "shelf_B1",
            "pick_pose_candidates": [{"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}],
            "place_pose_candidates": [{"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}],
            "constraints": {"clearance_m": 0.02},
            "reachability_hint": {"ik_feasible": True, "min_clearance_est": 0.02},
            "grasp_hint": {"pregrasp_offset": 0.08, "approach_axis": [0.0, 0.0, -1.0]},
            "subtask_graph": {"skill_mode": "APPROACH", "joint_trajectory": []},
        }
        with self.assertRaises(IntentValidationError):
            validate_intent_packet(bad_payload)

    def test_failure_path_missing_object(self):
        slot_map = SlotMap.from_dict(_slotmap_payload())
        with self.assertRaises(IntentResolutionError) as ctx:
            build_intent_packet(
                "MOVE_PLATE(shelf_A1, shelf_B1)",
                slot_map,
                [{"object_id": "tray1", "confidence": 0.2, "stamp_sec": 10.0}],
                now_sec=12.0,
            )
        self.assertEqual(ctx.exception.code, IntentFailureCode.MISSING_OBJECT)

    def test_failure_path_unreachable(self):
        payload = _slotmap_payload()
        payload["slots"][1]["allowed_objects"] = ["tray2"]
        slot_map = SlotMap.from_dict(payload)
        with self.assertRaises(IntentResolutionError) as ctx:
            build_intent_packet(
                "MOVE_PLATE(shelf_A1, shelf_B1)",
                slot_map,
                [{"object_id": "tray1", "confidence": 0.9, "stamp_sec": 10.0}],
                now_sec=10.1,
            )
        self.assertEqual(ctx.exception.code, IntentFailureCode.UNREACHABLE)

    def test_failure_path_disambiguation_required(self):
        payload = _slotmap_payload()
        payload["slots"].append(
            {
                "slot_id": "shelf_B2",
                "region_world": {"center_xyz": [-0.8, -1.16, 1.22], "size_xyz": [0.18, 0.18, 0.06], "yaw": 0.0},
                "approach_pose_candidates": [{"xyz": [-0.76, -1.10, 1.32], "rpy": [3.14, 0.0, 3.14]}],
                "place_pose_candidates": [{"xyz": [-0.80, -1.16, 1.22], "rpy": [3.14, 0.0, 3.14]}],
                "allowed_objects": ["tray1"],
                "priority": 1,
            }
        )
        slot_map = SlotMap.from_dict(payload)
        with self.assertRaises(IntentResolutionError) as ctx:
            build_intent_packet(
                "MOVE_PLATE(shelf_A1, shelf_B)",
                slot_map,
                [{"object_id": "tray1", "confidence": 0.9, "stamp_sec": 10.0}],
                now_sec=10.1,
            )
        self.assertEqual(ctx.exception.code, IntentFailureCode.TASK_DISAMBIGUATION_REQUIRED)

    def test_perception_adapter_switch_modes(self):
        phase0 = PerceptionAdapter(PerceptionAdapterConfig(mode="phase0_gt_proxy", min_confidence=0.1))
        out_phase0 = phase0.adapt(
            gt_proxy_objects=[{"object_id": "tray1", "xyz": [0.0, 0.0, 0.0], "stamp_sec": 5.0}],
            vision_objects=[{"object_id": "tray1", "xyz": [1.0, 1.0, 1.0], "confidence": 0.99, "stamp_sec": 5.0}],
            now_sec=5.1,
        )
        self.assertEqual(len(out_phase0), 1)
        self.assertEqual(out_phase0[0].xyz, (0.0, 0.0, 0.0))
        self.assertEqual(phase0.output_topic, "/v5/perception/object_pose_est")

        phase1 = PerceptionAdapter(PerceptionAdapterConfig(mode="phase1_vision_only", min_confidence=0.8))
        out_phase1 = phase1.adapt(
            gt_proxy_objects=[{"object_id": "tray1", "xyz": [0.0, 0.0, 0.0], "stamp_sec": 5.0}],
            vision_objects=[{"object_id": "tray1", "xyz": [1.0, 1.0, 1.0], "confidence": 0.9, "stamp_sec": 5.0}],
            now_sec=5.1,
        )
        self.assertEqual(len(out_phase1), 1)
        self.assertEqual(out_phase1[0].xyz, (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
