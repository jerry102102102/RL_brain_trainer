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


class _ResetFakeFuture:
    def __init__(self, *, done: bool, result_obj=None):
        self._done = done
        self._result_obj = result_obj

    def done(self):
        return self._done

    def result(self):
        return self._result_obj


class _ResetFakeGoalHandle:
    def __init__(self, *, accepted: bool, result_future: _ResetFakeFuture):
        self.accepted = accepted
        self._result_future = result_future

    def get_result_async(self):
        return self._result_future


class _ResetFakeActionClient:
    def __init__(self, *, wait_ok=True, send_future=None):
        self._wait_ok = wait_ok
        self._send_future = send_future

    def wait_for_server(self, timeout_sec):
        _ = timeout_sec
        return self._wait_ok

    def send_goal_async(self, goal):
        _ = goal
        return self._send_future


class _ResetFakeRclpy:
    def spin_until_future_complete(self, _node, _future, timeout_sec):
        _ = (_node, _future, timeout_sec)


class _ResetFakeJointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.header = types.SimpleNamespace(stamp=None)
        self.points = []


class _ResetFakeJointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = types.SimpleNamespace(sec=0)


class _ResetFakeGoal:
    def __init__(self):
        self.trajectory = None


class _ResetFakeFollowJointTrajectory:
    Goal = _ResetFakeGoal


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
        ex.reset_topic_fallback_attempts = 2
        ex.verbose_debug = False
        ex._canonical_joint_names_or_fail = lambda n_joints: tuple(f"joint_{i}" for i in range(n_joints))

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
        ex.reset_topic_fallback_attempts = 2
        ex.verbose_debug = False
        ex._canonical_joint_names_or_fail = lambda n_joints: tuple(f"joint_{i}" for i in range(n_joints))

        ex._send_reset_target = lambda **_kwargs: True
        ex._wait_reset_convergence = lambda **_kwargs: (False, np.full(7, 0.5), np.zeros(7), None)
        ex.read_runtime_state = lambda n_joints: (np.ones(n_joints, dtype=float), np.zeros(n_joints, dtype=float), None)
        ex._last_reset_action_diag = {"path": "action", "status": "result_timeout", "detail": "timeout=0.50"}

        with self.assertRaisesRegex(RuntimeError, "top_joint_errors") as ctx:
            ex.reset_arm_to_initial(n_joints=7, timeout_sec=0.5)
        self.assertIn("action_diag=path=action status=result_timeout", str(ctx.exception))

    def test_send_reset_target_action_requires_result_success(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex._JointTrajectory = _ResetFakeJointTrajectory
        ex._JointTrajectoryPoint = _ResetFakeJointTrajectoryPoint
        ex._FollowJointTrajectory = _ResetFakeFollowJointTrajectory
        ex._node = object()
        ex._rclpy = _ResetFakeRclpy()
        ex._traj_pub = types.SimpleNamespace(publish=lambda _msg: None)
        ex._canonical_joint_names_or_fail = lambda n_joints: tuple(f"joint_{i}" for i in range(n_joints))

        wrapped = types.SimpleNamespace(status=6, result=types.SimpleNamespace(error_code=1))
        result_future = _ResetFakeFuture(done=True, result_obj=wrapped)
        goal_handle = _ResetFakeGoalHandle(accepted=True, result_future=result_future)
        ex._traj_action_client = _ResetFakeActionClient(wait_ok=True, send_future=_ResetFakeFuture(done=True, result_obj=goal_handle))

        ok = ex._send_reset_target(n_joints=7, q_goal=np.zeros(7, dtype=float), timeout_sec=0.5, use_action=True)
        self.assertFalse(ok)
        self.assertEqual(ex._last_reset_action_diag.get("status"), "result_rejected")


if __name__ == "__main__":
    unittest.main()
