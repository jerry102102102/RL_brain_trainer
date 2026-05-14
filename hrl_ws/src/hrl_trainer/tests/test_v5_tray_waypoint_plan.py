from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.tray_waypoint_plan import default_tray_carry_waypoints, write_outputs


class TrayWaypointPlanTest(unittest.TestCase):
    def test_default_waypoints_are_controlled_sim_targets(self) -> None:
        waypoints = default_tray_carry_waypoints()
        self.assertGreaterEqual(len(waypoints), 4)
        for waypoint in waypoints:
            self.assertEqual(len(waypoint.pose6), 6)
            self.assertAlmostEqual(waypoint.rpy[0], 1.5708, places=4)
            self.assertAlmostEqual(waypoint.rpy[1], 0.0, places=4)
            target = waypoint.to_control_target()
            self.assertIn("pose6", target)
            self.assertEqual(target["source"], "vlm_semantic_waypoint/local_level_pose6")

    def test_write_outputs_creates_plan_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_outputs(
                output_dir=Path(tmp),
                instruction="Move tray1 while keeping it level.",
                source_slot="shelf_A1",
                target_slot="shelf_B1",
                object_id="tray1",
            )
            plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))
            targets = json.loads(Path(result["targets_path"]).read_text(encoding="utf-8"))
            self.assertEqual(plan["pipeline"], ["APPROACH", "FINISHER"])
            self.assertFalse(plan["safety_boundary"]["l1_outputs_joint_trajectory"])
            self.assertEqual(len(plan["waypoints"]), len(targets["targets"]))
            self.assertEqual(targets["target_encoding"], "pose6")


if __name__ == "__main__":
    unittest.main()
