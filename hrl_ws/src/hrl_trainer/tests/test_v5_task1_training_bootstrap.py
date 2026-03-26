import unittest

import numpy as np

from hrl_trainer.v5.task1_train import (
    L2Action,
    Task1Config,
    Task1State,
    L3ExecutionResult,
    HighPoseTargetProvider,
    run_task1_episode,
    adapt_action_delta_q,
    build_task1_observation,
    check_done_success,
    compose_task1_reward,
)


class TestV5Task1TrainingBootstrap(unittest.TestCase):
    def _build_state(self) -> Task1State:
        return Task1State(
            q=np.array([0.1, -0.1, 0.3, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(6, dtype=float),
            target_pose_xyz=np.array([0.2, 0.0, 0.35], dtype=float),
            step=5,
            max_steps=40,
            safe_z_min=0.2,
        )

    def test_observation_schema_shape(self):
        obs = build_task1_observation(self._build_state())
        self.assertEqual(obs.q.shape, (6,))
        self.assertEqual(obs.dq.shape, (6,))
        self.assertEqual(obs.delta_p.shape, (3,))
        self.assertGreaterEqual(obs.t_remain, 0.0)
        self.assertLessEqual(obs.t_remain, 1.0)

    def test_reward_mode_behavior_differs(self):
        cfg = Task1Config()
        s0 = self._build_state()
        s1 = Task1State(
            q=np.array([0.12, -0.08, 0.31, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(6, dtype=float),
            target_pose_xyz=s0.target_pose_xyz,
            step=s0.step + 1,
            max_steps=s0.max_steps,
            safe_z_min=s0.safe_z_min,
        )
        o0 = build_task1_observation(s0)
        o1 = build_task1_observation(s1)
        cmd = np.array([0.02, 0.02, 0.01, 0, 0, 0], dtype=float)

        no_shaping = compose_task1_reward(
            mode="no_shaping",
            obs_prev=o0,
            obs_next=o1,
            delta_q_cmd=cmd,
            safety_violation=0.0,
            sat_ratio=0.9,
            no_motion=False,
            done=False,
            success=False,
            cfg=cfg,
        )
        task1_main = compose_task1_reward(
            mode="task1_main",
            obs_prev=o0,
            obs_next=o1,
            delta_q_cmd=cmd,
            safety_violation=0.0,
            sat_ratio=0.9,
            no_motion=False,
            done=False,
            success=False,
            cfg=cfg,
        )

        self.assertNotEqual(no_shaping, task1_main)

    def test_feasibility_penalty_toggle_affects_reward(self):
        s0 = self._build_state()
        s1 = Task1State(
            q=np.array([0.12, -0.08, 0.31, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(6, dtype=float),
            target_pose_xyz=s0.target_pose_xyz,
            step=s0.step + 1,
            max_steps=s0.max_steps,
            safe_z_min=s0.safe_z_min,
        )
        o0 = build_task1_observation(s0)
        o1 = build_task1_observation(s1)
        cmd = np.array([0.02, 0.02, 0.01, 0, 0, 0], dtype=float)

        cfg_off = Task1Config(enable_feasibility_penalty=False, lambda_inf=2.0, lambda_rep=2.0, lambda_sat=2.0)
        cfg_on = Task1Config(enable_feasibility_penalty=True, lambda_inf=2.0, lambda_rep=2.0, lambda_sat=2.0)

        reward_off = compose_task1_reward(
            mode="task1_main",
            obs_prev=o0,
            obs_next=o1,
            delta_q_cmd=cmd,
            safety_violation=0.0,
            sat_ratio=0.8,
            no_motion=False,
            done=False,
            success=False,
            cfg=cfg_off,
            feasible_ratio=0.2,
            projection_gap=0.5,
            null_effect_step=True,
        )
        reward_on = compose_task1_reward(
            mode="task1_main",
            obs_prev=o0,
            obs_next=o1,
            delta_q_cmd=cmd,
            safety_violation=0.0,
            sat_ratio=0.8,
            no_motion=False,
            done=False,
            success=False,
            cfg=cfg_on,
            feasible_ratio=0.2,
            projection_gap=0.5,
            null_effect_step=True,
        )
        reward_off_zero = compose_task1_reward(
            mode="task1_main",
            obs_prev=o0,
            obs_next=o1,
            delta_q_cmd=cmd,
            safety_violation=0.0,
            sat_ratio=0.8,
            no_motion=False,
            done=False,
            success=False,
            cfg=Task1Config(enable_feasibility_penalty=False, lambda_inf=0.0, lambda_rep=0.0, lambda_sat=0.0),
            feasible_ratio=0.2,
            projection_gap=0.5,
            null_effect_step=True,
        )

        self.assertAlmostEqual(reward_off, reward_off_zero, places=9)
        self.assertLess(reward_on, reward_off)
        self.assertGreater(abs(reward_off - reward_on), 0.1)

    def test_done_success_conditions(self):
        cfg = Task1Config(success_pos_tol=0.05)
        state = self._build_state()

        near_state = Task1State(
            q=np.array([0.2, 0.0, 0.36, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(6, dtype=float),
            target_pose_xyz=state.target_pose_xyz,
            step=10,
            max_steps=40,
            safe_z_min=0.2,
        )
        done, success, reason = check_done_success(near_state, build_task1_observation(near_state), safety_violation=0.0, cfg=cfg)
        self.assertTrue(done)
        self.assertTrue(success)
        self.assertEqual(reason, "success")

        timeout_state = Task1State(
            q=state.q,
            dq=state.dq,
            target_pose_xyz=state.target_pose_xyz,
            step=40,
            max_steps=40,
            safe_z_min=0.2,
        )
        done, success, reason = check_done_success(timeout_state, build_task1_observation(timeout_state), safety_violation=0.0, cfg=cfg)
        self.assertTrue(done)
        self.assertFalse(success)
        self.assertEqual(reason, "timeout")

    def test_action_adapter_bounds(self):
        a_raw, bounded = adapt_action_delta_q(
            L2Action(delta_q_raw=np.array([3.0, -3.0, 0.04, 0, 0, 0])),
            n_joints=6,
            dq_max_per_joint=np.full(6, 0.05, dtype=float),
        )
        self.assertTrue(np.all(np.abs(a_raw) <= 1.0 + 1e-9))
        self.assertTrue(np.all(np.abs(bounded) <= 0.05 + 1e-9))
        self.assertAlmostEqual(float(a_raw[0]), float(np.tanh(3.0)), places=6)
        self.assertAlmostEqual(float(a_raw[1]), float(np.tanh(-3.0)), places=6)

    def test_no_effect_action_does_not_count_as_saturation_rate(self):
        cfg = Task1Config(max_steps=1, n_joints=6)

        class _FixedL2:
            def decide_action(self, _obs):
                return L2Action(delta_q_raw=np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float))

        class _NoEffectL3:
            def execute_with_safety(self, state, _delta_q_cmd):
                return L3ExecutionResult(
                    accepted=False,
                    q_next=state.q.copy(),
                    dq_next=np.zeros_like(state.q),
                    safety_violation=0.0,
                    ee_proxy_xyz=state.ee_proxy_xyz,
                    requested_delta_q=np.array([0.02, 0, 0, 0, 0, 0], dtype=float),
                    executed_delta_q=np.zeros(6, dtype=float),
                    feasible_ratio=0.0,
                    projection_gap=0.02,
                    null_effect_step=True,
                    sat_ratio=1.0,
                    no_effect_action=True,
                    logs=("L3_EXEC:no_effect_action_readback_stale",),
                )

        row = run_task1_episode(
            episode_index=0,
            reward_mode="task1_main",
            cfg=cfg,
            dq_max_per_joint=np.full(6, 0.05, dtype=float),
            l1_provider=HighPoseTargetProvider(target_xyz=np.array([0.2, 0.0, 0.35], dtype=float)),
            l2_policy=_FixedL2(),
            l3_executor=_NoEffectL3(),
            initial_q=np.array([0.1, -0.1, 0.3, 0.0, 0.0, 0.0], dtype=float),
            initial_dq=np.zeros(6, dtype=float),
            initial_ee_proxy_xyz=np.array([0.1, -0.1, 0.3], dtype=float),
        )

        self.assertEqual(row["episode_summary"]["saturation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
