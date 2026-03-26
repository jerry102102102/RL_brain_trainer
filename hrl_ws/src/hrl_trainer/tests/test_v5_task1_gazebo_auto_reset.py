import types
import unittest
from unittest.mock import patch

import numpy as np

from hrl_trainer.v5.task1_train import GazeboRuntimeL3Executor, Task1Config, run_task1_training


class _FakeGazeboExecutorOk:
    reset_calls = 0

    def __init__(self, **_kwargs):
        self._n_joints = 6

    def close(self) -> None:
        return

    def ee_source_name(self) -> str:
        return "fake_ee_source"

    def runtime_joint_names(self, n_joints: int):
        return tuple(f"joint_{i}" for i in range(n_joints))

    def read_runtime_state(self, n_joints: int):
        self._n_joints = n_joints
        q = np.zeros(n_joints, dtype=float)
        q[2] = 0.22
        return q, np.zeros(n_joints, dtype=float), q[:3].copy()

    def execute_with_safety(self, state, delta_q_cmd):
        dq = np.asarray(delta_q_cmd, dtype=float)
        return types.SimpleNamespace(
            accepted=True,
            q_next=state.q.copy(),
            dq_next=np.zeros_like(state.q),
            safety_violation=0.0,
            ee_proxy_xyz=state.ee_proxy_xyz.copy() if state.ee_proxy_xyz is not None else state.q[:3].copy(),
            logs=("L3_EXEC:path=fake_gazebo",),
            limited_cmd=dq.copy(),
            q_target_minus_runtime=dq.copy(),
            requested_delta_q=dq.copy(),
            executed_delta_q=np.zeros_like(state.q),
            feasible_ratio=1.0,
            projection_gap=0.0,
            null_effect_step=False,
            sat_ratio=0.0,
            encoder_delta=0.0,
            no_motion_signal=False,
            no_effect_action=False,
        )

    def reset_episode(self, *, n_joints: int, timeout_sec: float | None = None):
        _ = timeout_sec
        type(self).reset_calls += 1
        q, dq, ee = self.read_runtime_state(n_joints=n_joints)
        return {
            "applied": True,
            "success": True,
            "duration_ms": 1.0,
            "initial_state": {"q": q.tolist(), "dq": dq.tolist(), "ee_proxy": ee.tolist()},
            "error": None,
        }


class _FakeGazeboExecutorFail(_FakeGazeboExecutorOk):
    def reset_episode(self, *, n_joints: int, timeout_sec: float | None = None):
        _ = (n_joints, timeout_sec)
        type(self).reset_calls += 1
        err = RuntimeError("synthetic reset failure")
        setattr(
            err,
            "reset_meta",
            {"applied": True, "success": False, "duration_ms": 0.5, "initial_state": None, "error": "synthetic reset failure"},
        )
        raise err


class TestTask1GazeboAutoReset(unittest.TestCase):
    def test_gazebo_path_invokes_reset_between_episodes(self):
        _FakeGazeboExecutorOk.reset_calls = 0
        with patch("hrl_trainer.v5.task1_train.GazeboRuntimeL3Executor", _FakeGazeboExecutorOk):
            rows = run_task1_training(
                episodes=3,
                reward_mode="heuristic",
                cfg=Task1Config(max_steps=1),
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(_FakeGazeboExecutorOk.reset_calls, 2)
        self.assertIn("reset", rows[0])
        self.assertFalse(rows[0]["reset"]["applied"])
        self.assertIn("reset", rows[1])
        self.assertTrue(rows[1]["reset"]["applied"])

    def test_gazebo_reset_failure_raises_fail_fast(self):
        _FakeGazeboExecutorFail.reset_calls = 0
        with patch("hrl_trainer.v5.task1_train.GazeboRuntimeL3Executor", _FakeGazeboExecutorFail):
            with self.assertRaisesRegex(RuntimeError, "episode reset failed before episode 1"):
                run_task1_training(
                    episodes=2,
                    reward_mode="heuristic",
                    cfg=Task1Config(max_steps=1),
                )

        self.assertEqual(_FakeGazeboExecutorFail.reset_calls, 1)

    def test_reset_arm_to_initial_uses_topic_fallback_when_action_stalls(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex.q_min = np.array([-1.0] * 7, dtype=float)
        ex.q_max = np.array([1.0] * 7, dtype=float)
        ex.reset_position_tol = 0.01
        ex.verbose_debug = False

        action_calls = []

        def _send_reset_target(*, n_joints: int, q_goal: np.ndarray, timeout_sec: float, use_action: bool) -> bool:
            _ = (n_joints, q_goal, timeout_sec)
            action_calls.append(use_action)
            return use_action

        attempts = [
            (False, np.ones(7, dtype=float), np.zeros(7, dtype=float), None),
            (True, np.zeros(7, dtype=float), np.zeros(7, dtype=float), np.zeros(3, dtype=float)),
        ]

        ex._send_reset_target = _send_reset_target
        ex._wait_reset_convergence = lambda **_kwargs: attempts.pop(0)
        ex.read_runtime_state = lambda n_joints: (np.ones(n_joints, dtype=float), np.zeros(n_joints, dtype=float), None)

        q, dq, ee = ex.reset_arm_to_initial(n_joints=7, timeout_sec=0.5)

        self.assertEqual(action_calls, [True, False])
        self.assertTrue(np.allclose(q, np.zeros(7, dtype=float)))
        self.assertTrue(np.allclose(dq, np.zeros(7, dtype=float)))
        self.assertTrue(np.allclose(ee, np.zeros(3, dtype=float)))

    def test_reset_arm_to_initial_fail_fast_when_fallback_also_fails(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex.q_min = np.array([-1.0] * 7, dtype=float)
        ex.q_max = np.array([1.0] * 7, dtype=float)
        ex.reset_position_tol = 0.01
        ex.verbose_debug = False

        ex._send_reset_target = lambda **_kwargs: True
        ex._wait_reset_convergence = lambda **_kwargs: (False, np.full(7, 0.5), np.zeros(7), None)
        ex.read_runtime_state = lambda n_joints: (np.ones(n_joints, dtype=float), np.zeros(n_joints, dtype=float), None)

        with self.assertRaisesRegex(RuntimeError, "arm reset did not converge"):
            ex.reset_arm_to_initial(n_joints=7, timeout_sec=0.5)


if __name__ == "__main__":
    unittest.main()
