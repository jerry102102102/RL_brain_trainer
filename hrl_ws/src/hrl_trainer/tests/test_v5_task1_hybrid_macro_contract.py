import types
import unittest

import numpy as np

from hrl_trainer.v5.task1_train import GazeboRuntimeL3Executor, Task1Config, Task1State, run_task1_training


class _FakeFuture:
    def __init__(self, result_obj=None):
        self._result = result_obj

    def done(self):
        return True

    def result(self):
        return self._result


class _FakeActionClient:
    def __init__(self):
        self.send_calls = 0

    def wait_for_server(self, timeout_sec):
        _ = timeout_sec
        return True

    def send_goal_async(self, _goal):
        self.send_calls += 1
        return _FakeFuture(result_obj=types.SimpleNamespace(accepted=True, get_result_async=lambda: _FakeFuture(types.SimpleNamespace(status=4))))


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


class TestHybridMacroContract(unittest.TestCase):
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

    def test_hybrid_routing_uses_action_for_reset_and_topic_for_micro_step(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex.max_dq_per_step = 0.05
        ex.action_timeout_sec = 0.8
        ex.action_server_timeout_sec = 0.3
        ex.action_min_motion_tol = 1e-4
        ex.action_name = "/arm_controller/follow_joint_trajectory"
        ex.l3_exec_mode = "hybrid"
        ex.macro_kickoff_action = True
        ex.q_min = np.array([-1.0] * 7, dtype=float)
        ex.q_max = np.array([1.0] * 7, dtype=float)
        ex._latest_joint_state = types.SimpleNamespace(name=[f"j{i}" for i in range(7)])
        ex._JointTrajectory = _FakeJointTrajectory
        ex._JointTrajectoryPoint = _FakeJointTrajectoryPoint
        ex._FollowJointTrajectory = _FakeFollowJointTrajectory
        ex._node = types.SimpleNamespace(get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: object())))
        ex._rclpy = _FakeRclpy()
        ex._traj_pub = types.SimpleNamespace(publish=lambda _msg: None)
        ex._spin_once = lambda _timeout=0.05: None
        ex._latest_joint_stamp_ns = 1
        ex._traj_action_client = _FakeActionClient()
        ex.ee_source_name = lambda: "ee_pose_topic"
        ex.reset_position_tol = 0.03

        q0 = np.array([0.0, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        q1 = np.array([0.01, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        q2 = np.array([0.02, 0.0, 0.25, 0, 0, 0, 0], dtype=float)
        calls = [
            (q0, np.zeros(7), q0[:3]),  # reset initial read
            (q0, np.zeros(7), q0[:3]),  # reset convergence read
            (q0, np.zeros(7), q0[:3]),  # kickoff pre-read
            (q1, np.zeros(7), q1[:3]),  # kickoff finalize read
            (q1, np.zeros(7), q1[:3]),  # micro pre-read
            (q2, np.zeros(7), q2[:3]),  # micro finalize read
        ]
        ex.read_runtime_state = lambda n_joints: calls.pop(0)

        ex.reset_arm_to_initial(n_joints=7, timeout_sec=0.2, target_q=q0)
        kickoff = ex.execute_macro_kickoff(self._state(), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=float))
        step = ex.execute_with_safety(self._state(), np.array([0.01, 0, 0, 0, 0, 0, 0], dtype=float))

        self.assertTrue(any("L3_EXEC:mode=action" in s for s in kickoff.logs))
        self.assertTrue(any("L3_EXEC:mode=topic" in s for s in step.logs))
        self.assertGreaterEqual(ex._traj_action_client.send_calls, 2)

    def test_decision_id_propagates_to_artifacts(self):
        rows = run_task1_training(
            episodes=1,
            reward_mode="heuristic",
            backend="bootstrap",
            cfg=Task1Config(max_steps=4),
            l3_exec_mode="hybrid",
            macro_ttl_steps=2,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("replay_by_decision", row)
        self.assertGreaterEqual(len(row["replay_by_decision"]), 1)
        self.assertIn("episode_summary", row)
        self.assertIn("macro_decisions", row["episode_summary"])
        for step in row["trajectory"]:
            self.assertIn("decision_id", step)
            self.assertIn("state_version", step)
            self.assertIn("decision_ttl", step)


if __name__ == "__main__":
    unittest.main()
