import unittest

import numpy as np

from hrl_trainer.kinematic_phase1.kinematics.joint_limits import default_joint_specs, lower_bounds, upper_bounds
from hrl_trainer.v5.phase3a_controlled_sim import action_to_command_q, build_default_targets, pose_error_norms


class TestV5Phase3AControlledSim(unittest.TestCase):
    def test_action_to_command_q_clips_to_limits(self):
        specs = default_joint_specs()
        q = upper_bounds(specs) - 1e-4
        cmd = action_to_command_q(q=q, action=np.ones(7), joint_specs=specs, action_delta_scale=1.0)
        self.assertTrue(np.all(cmd <= upper_bounds(specs)))
        self.assertTrue(np.all(cmd >= lower_bounds(specs)))

    def test_default_targets_are_reachable_fk_targets(self):
        specs = default_joint_specs()
        q = np.zeros(7, dtype=float)
        targets = build_default_targets(q_reference=q, joint_specs=specs, max_targets=2)
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertEqual(target.goal_pose6.shape, (6,))
            self.assertIsNotNone(target.q_goal)
            self.assertEqual(target.source, "default_q_delta_fk:smoke")

    def test_visible_workspace_targets_cover_twenty_fk_targets(self):
        specs = default_joint_specs()
        q = np.zeros(7, dtype=float)
        targets = build_default_targets(q_reference=q, joint_specs=specs, max_targets=20, profile="visible_workspace")
        self.assertEqual(len(targets), 20)
        self.assertTrue(all("visible_workspace" in target.source for target in targets))
        self.assertGreater(max(float(np.linalg.norm(target.q_goal - q)) for target in targets), 0.25)

    def test_pose_error_norms_returns_zero_for_same_pose(self):
        pose = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
        pos, ori = pose_error_norms(pose, pose)
        self.assertAlmostEqual(pos, 0.0)
        self.assertAlmostEqual(ori, 0.0)


if __name__ == "__main__":
    unittest.main()
