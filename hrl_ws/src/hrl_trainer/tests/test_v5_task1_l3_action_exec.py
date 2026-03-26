import types
import unittest

import numpy as np

from hrl_trainer.v5.task1_train import GazeboRuntimeL3Executor, Task1Config, Task1State, run_task1_training


class _FakeFuture:
    def __init__(self, *, done: bool, result_obj=None):
        self._done = done
        self._result = result_obj

    def done(self):
        return self._done

    def result(self):
        return self._result


class _FakeGoalHandle:
    def __init__(self, *, accepted: bool, result_future: _FakeFuture):
        self.accepted = accepted
        self._result_future = result_future

    def get_result_async(self):
        return self._result_future


class _FakeActionClient:
    def __init__(self, *, wait_ok=True, send_future=None):
        self._wait_ok = wait_ok
        self._send_future = send_future

    def wait_for_server(self, timeout_sec):
        _ = timeout_sec
        return self._wait_ok

    def send_goal_async(self, _goal):
        return self._send_future


class _FakeRclpy:
    def spin_until_future_complete(self, _node, _future, timeout_sec):
        _ = timeout_sec


class _FakeJointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.header = types.SimpleNamespace(stamp=None)
        self.points = []


class _FakeJointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = types.SimpleNamespace(sec=0)


class _FakeGoal:
    def __init__(self):
        self.trajectory = None


class _FakeFollowJointTrajectory:
    Goal = _FakeGoal


class TestTask1L3ActionExecution(unittest.TestCase):
    def _build_executor(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex.max_dq_per_step = 0.05
        ex.action_timeout_sec = 0.8
        ex.action_server_timeout_sec = 0.3
        ex.action_min_motion_tol = 1e-4
        ex.action_name = "/arm_controller/follow_joint_trajectory"
        ex.action_post_result_wait_sec = 0.01
        ex.l3_exec_mode = "action"
        ex.q_min = np.array([-1.0] * 7, dtype=float)
        ex.q_max = np.array([1.0] * 7, dtype=float)
        ex._latest_joint_state = types.SimpleNamespace(name=[f"j{i}" for i in range(7)])
        ex._fk_solver = None
        ex._JointTrajectory = _FakeJointTrajectory
        ex._JointTrajectoryPoint = _FakeJointTrajectoryPoint
        ex._FollowJointTrajectory = _FakeFollowJointTrajectory
        ex._node = types.SimpleNamespace(get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: object())))
        ex._rclpy = _FakeRclpy()
        ex._latest_joint_stamp_ns = 1
        ex._spin_once = lambda _timeout=0.05: None
        ex.ee_source_name = lambda: "ee_pose_topic"
        ex.verbose_debug = False
        return ex

    def _state(self):
        q = np.zeros(7, dtype=float)
        q[2] = 0.25
        return Task1State(
            q=q,
            dq=np.zeros(7, dtype=float),
            target_pose_xyz=np.array([0.35, 0.0, 0.35], dtype=float),
            step=0,
            max_steps=40,
            safe_z_min=0.20,
            ee_proxy_xyz=np.array([0.0, 0.0, 0.25], dtype=float),
        )

    def test_action_client_success_path(self):
        ex = self._build_executor()
        result_future = _FakeFuture(done=True, result_obj=types.SimpleNamespace(status=4, result=object()))
        goal_handle = _FakeGoalHandle(accepted=True, result_future=result_future)
        ex._traj_action_client = _FakeActionClient(wait_ok=True, send_future=_FakeFuture(done=True, result_obj=goal_handle))

        q0 = np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        q1 = np.array([0.02, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        calls = [(q0, np.zeros(7), np.array([0.0, 0.0, 0.25])), (q1, np.zeros(7), np.array([0.02, 0.0, 0.25]))]
        ex.read_runtime_state = lambda n_joints: calls.pop(0)

        out = ex.execute_with_safety(self._state(), np.array([0.02, 0, 0, 0, 0, 0, 0], dtype=float))
        self.assertTrue(out.accepted)
        self.assertTrue(any("action_goal_response=accepted" in s for s in out.logs))
        self.assertTrue(any("action_result_status=4" in s for s in out.logs))
        self.assertTrue(any("post_action_joint_delta_max=" in s for s in out.logs))

    def test_action_goal_rejected_path(self):
        ex = self._build_executor()
        goal_handle = _FakeGoalHandle(accepted=False, result_future=_FakeFuture(done=True, result_obj=types.SimpleNamespace(status=6)))
        ex._traj_action_client = _FakeActionClient(wait_ok=True, send_future=_FakeFuture(done=True, result_obj=goal_handle))
        ex.read_runtime_state = lambda n_joints: (np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float), np.zeros(7), np.array([0.0, 0.0, 0.25]))

        out = ex.execute_with_safety(self._state(), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=float))
        self.assertFalse(out.accepted)
        self.assertTrue(any("action_goal_response=rejected" in s for s in out.logs))

    def test_action_result_timeout_path(self):
        ex = self._build_executor()
        goal_handle = _FakeGoalHandle(accepted=True, result_future=_FakeFuture(done=False, result_obj=None))
        ex._traj_action_client = _FakeActionClient(wait_ok=True, send_future=_FakeFuture(done=True, result_obj=goal_handle))
        ex.read_runtime_state = lambda n_joints: (np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float), np.zeros(7), np.array([0.0, 0.0, 0.25]))

        out = ex.execute_with_safety(self._state(), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=float))
        self.assertFalse(out.accepted)
        self.assertTrue(any("action_result_timeout=" in s for s in out.logs))

    def test_action_status_success_but_error_code_marks_rejected(self):
        ex = self._build_executor()
        wrapped = types.SimpleNamespace(error_code=1, error_string="INVALID_GOAL")
        result_future = _FakeFuture(done=True, result_obj=types.SimpleNamespace(status=4, result=wrapped))
        goal_handle = _FakeGoalHandle(accepted=True, result_future=result_future)
        ex._traj_action_client = _FakeActionClient(wait_ok=True, send_future=_FakeFuture(done=True, result_obj=goal_handle))
        ex.read_runtime_state = lambda n_joints: (np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float), np.zeros(7), np.array([0.0, 0.0, 0.25]))

        out = ex.execute_with_safety(self._state(), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=float))
        self.assertFalse(out.accepted)
        self.assertTrue(any("action_result_error_code=1" in s for s in out.logs))

    def test_action_no_effect_readback_stale_is_tagged_and_not_saturated(self):
        ex = self._build_executor()
        wrapped = types.SimpleNamespace(error_code=0, error_string="")
        result_future = _FakeFuture(done=True, result_obj=types.SimpleNamespace(status=4, result=wrapped))
        goal_handle = _FakeGoalHandle(accepted=True, result_future=result_future)
        ex._traj_action_client = _FakeActionClient(wait_ok=True, send_future=_FakeFuture(done=True, result_obj=goal_handle))
        q0 = np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        ex.read_runtime_state = lambda n_joints: (q0.copy(), np.zeros(7), q0[:3].copy())

        out = ex.execute_with_safety(self._state(), np.array([0.02, 0, 0, 0, 0, 0, 0], dtype=float))
        self.assertFalse(out.accepted)
        self.assertTrue(out.no_effect_action)
        self.assertAlmostEqual(out.sat_ratio, 0.0)
        self.assertTrue(any("L3_EXEC:no_effect_action_readback_stale" in s for s in out.logs))

    def test_gazebo_default_mode_resolves_to_action(self):
        captured = {}

        class _FakeGazeboExecutor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def close(self):
                return

            def ee_source_name(self):
                return "fake"

            def runtime_joint_names(self, n_joints):
                return tuple(f"joint_{i}" for i in range(n_joints))

            def read_runtime_state(self, n_joints):
                q = np.zeros(n_joints, dtype=float)
                q[2] = 0.25
                return q, np.zeros(n_joints, dtype=float), q[:3].copy()

            def execute_with_safety(self, state, _delta_q_cmd):
                from hrl_trainer.v5.task1_train import L3ExecutionResult

                return L3ExecutionResult(
                    accepted=True,
                    q_next=state.q.copy(),
                    dq_next=np.zeros_like(state.q),
                    safety_violation=0.0,
                    ee_proxy_xyz=state.ee_proxy_xyz,
                    logs=("ok",),
                    requested_delta_q=np.zeros_like(state.q),
                    executed_delta_q=np.zeros_like(state.q),
                )

        from unittest.mock import patch

        with patch("hrl_trainer.v5.task1_train.GazeboRuntimeL3Executor", _FakeGazeboExecutor):
            run_task1_training(episodes=1, reward_mode="heuristic", cfg=Task1Config(max_steps=1))

        self.assertIn("action_timeout_sec", captured)


if __name__ == "__main__":
    unittest.main()
