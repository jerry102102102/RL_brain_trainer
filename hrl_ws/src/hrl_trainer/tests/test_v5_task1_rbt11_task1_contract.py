import unittest

import numpy as np

from hrl_trainer.v5.task1_train import (
    SafetyConstrainedL3Executor,
    Task1Config,
    Task1Observation,
    Task1State,
    compose_task1_reward,
    resolve_dq_max_per_joint,
)


class TestRBT11Task1Contract(unittest.TestCase):
    def test_delta_q_default_limits_prioritize_j2(self):
        cfg = Task1Config(n_joints=7)
        limits = resolve_dq_max_per_joint(cfg=cfg, n_joints=7)
        self.assertAlmostEqual(float(limits[0]), 0.03)
        self.assertAlmostEqual(float(limits[1]), 0.02)
        self.assertTrue(np.allclose(limits[[2, 3, 4, 5, 6]], 0.03))

    def test_l3_emits_no_motion_streak_signal(self):
        ex = SafetyConstrainedL3Executor(
            dq_max_per_joint=np.array([0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=float),
            epsilon_motion=0.002,
            stuck_window=3,
        )
        state = Task1State(
            q=np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(7, dtype=float),
            target_pose_xyz=np.array([0.35, 0.0, 0.35], dtype=float),
            step=0,
            max_steps=10,
            safe_z_min=0.2,
        )
        out0 = ex.execute_with_safety(state, np.zeros(7, dtype=float))
        out1 = ex.execute_with_safety(state, np.zeros(7, dtype=float))
        out2 = ex.execute_with_safety(state, np.zeros(7, dtype=float))
        self.assertFalse(any("NO_MOTION_STREAK" in s for s in out0.logs))
        self.assertFalse(any("NO_MOTION_STREAK" in s for s in out1.logs))
        self.assertTrue(any("NO_MOTION_STREAK" in s for s in out2.logs))

    def test_reward_only_progress_saturation_and_no_motion(self):
        cfg = Task1Config(
            saturation_threshold=0.95,
            reward_w_progress=1.0,
            reward_w_sat=-0.3,
            reward_w_nomotion=-0.8,
        )
        obs_prev = Task1Observation(
            q=np.zeros(7, dtype=float),
            dq=np.zeros(7, dtype=float),
            delta_p=np.array([0.1, 0.0, 0.0], dtype=float),
            d_pos=0.1,
            t_remain=1.0,
            z_margin=0.1,
        )
        obs_next = Task1Observation(
            q=np.zeros(7, dtype=float),
            dq=np.zeros(7, dtype=float),
            delta_p=np.array([0.08, 0.0, 0.0], dtype=float),
            d_pos=0.08,
            t_remain=0.9,
            z_margin=0.1,
        )
        reward = compose_task1_reward(
            mode="task1_main",
            obs_prev=obs_prev,
            obs_next=obs_next,
            delta_q_cmd=np.zeros(7, dtype=float),
            safety_violation=0.0,
            sat_ratio=1.0,
            no_motion=True,
            done=False,
            success=False,
            cfg=cfg,
        )
        # progress=+0.02, sat=1.0->penalty -0.3, no_motion penalty -0.8 => total -1.08
        self.assertAlmostEqual(float(reward), -1.08, places=6)


if __name__ == "__main__":
    unittest.main()
