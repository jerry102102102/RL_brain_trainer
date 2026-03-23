import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from hrl_trainer.v5 import task1_train
from hrl_trainer.v5.task1_train import GazeboRuntimeL3Executor, Task1Config, Task1State, run_task1_training


class _FakeBootstrapExecutor:
    def execute_with_safety(self, state, _delta_q_cmd):
        return type(
            "FakeExecResult",
            (),
            {
                "accepted": True,
                "q_next": state.q.copy(),
                "dq_next": np.zeros_like(state.q),
                "safety_violation": 0.0,
                "ee_proxy_xyz": state.ee_proxy_xyz,
                "logs": ("L3_EXEC:path=fake_bootstrap",),
            },
        )()


class TestGazeboEeSource(unittest.TestCase):
    def _build_executor(self) -> GazeboRuntimeL3Executor:
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex._latest_ee_pose_xyz = None
        ex._latest_joint_state = None
        ex.allow_ee_fallback = False
        ex.ee_pose_topic = "/ee_pose"
        ex.ee_base_frame = "base_link"
        ex.ee_tip_frame = "tool0"
        ex._lookup_ee_pose_from_tf = lambda: None
        ex._spin_once = lambda _timeout=0.1: None
        ex.verbose_debug = False
        ex._legacy_fk_joint_order = (
            "Rack_joint",
            "robot_base_joint",
            "shoulder1_joint",
            "shoulder2_joint",
            "wr1_joint",
            "wr2_joint",
            "wr3_joint",
        )
        ex._fk_solver = None
        return ex

    def test_ee_source_prefers_topic_then_tf_then_fk(self):
        ex = self._build_executor()
        ex._latest_ee_pose_xyz = np.array([0.2, 0.1, 0.4], dtype=float)
        ee, source = ex._resolve_ee_pose_or_fail(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(source, "ee_pose_topic")
        self.assertAlmostEqual(float(ee[2]), 0.4)

        ex._latest_ee_pose_xyz = None
        ex._lookup_ee_pose_from_tf = lambda: np.array([0.3, 0.1, 0.5], dtype=float)
        ee, source = ex._resolve_ee_pose_or_fail(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(source, "tf_lookup")
        self.assertAlmostEqual(float(ee[2]), 0.5)

        ex._lookup_ee_pose_from_tf = lambda: None
        ex._lookup_ee_pose_from_fk_joint_state = lambda: np.array([0.4, 0.2, 0.6], dtype=float)
        ee, source = ex._resolve_ee_pose_or_fail(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(source, "fk_joint_state")
        self.assertAlmostEqual(float(ee[2]), 0.6)

    def test_fk_provider_basic_input_output(self):
        ex = self._build_executor()
        ex._fk_solver = lambda q: np.array(
            [
                [1.0, 0.0, 0.0, float(q[0])],
                [0.0, 1.0, 0.0, float(q[1])],
                [0.0, 0.0, 1.0, float(q[2])],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        ex._latest_joint_state = type(
            "FakeJointState",
            (),
            {
                "name": list(ex._legacy_fk_joint_order),
                "position": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            },
        )()
        ee = ex._lookup_ee_pose_from_fk_joint_state()
        self.assertIsNotNone(ee)
        self.assertTrue(np.allclose(ee, np.array([0.1, 0.2, 0.3], dtype=float)))

    def test_read_runtime_state_uses_full_joint_state_for_fk_even_when_n_joints_is_6(self):
        ex = self._build_executor()
        fk_calls: list[np.ndarray] = []

        def _fake_fk(q):
            fk_calls.append(np.asarray(q, dtype=float).copy())
            return np.array(
                [
                    [1.0, 0.0, 0.0, float(q[0])],
                    [0.0, 1.0, 0.0, float(q[1])],
                    [0.0, 0.0, 1.0, float(q[2])],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        ex._fk_solver = _fake_fk
        ex._latest_joint_state = type(
            "FakeJointState",
            (),
            {
                "name": list(ex._legacy_fk_joint_order),
                "position": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77],
                "velocity": [0.0] * 7,
            },
        )()

        q, dq, ee = ex.read_runtime_state(n_joints=6)

        self.assertEqual(q.shape[0], 6)
        self.assertEqual(dq.shape[0], 6)
        self.assertIsNotNone(ee)
        self.assertTrue(np.allclose(ee, np.array([0.11, 0.22, 0.33], dtype=float)))
        self.assertEqual(len(fk_calls), 1)
        self.assertEqual(fk_calls[0].shape[0], 7)
        self.assertTrue(np.allclose(fk_calls[0], np.array([0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77], dtype=float)))

    def test_fk_skip_logs_debug_when_required_joint_names_missing(self):
        ex = self._build_executor()
        ex.verbose_debug = True
        ex._fk_solver = lambda q: np.eye(4, dtype=float)
        ex._latest_joint_state = type(
            "FakeJointState",
            (),
            {
                "name": ["Rack_joint", "robot_base_joint", "shoulder1_joint"],
                "position": [0.1, 0.2, 0.3],
            },
        )()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ee = ex._lookup_ee_pose_from_fk_joint_state()

        self.assertIsNone(ee)
        self.assertIn("fk_skip: required FK joint names missing", buf.getvalue())

    def test_fail_fast_without_ee_source(self):
        ex = self._build_executor()
        with self.assertRaisesRegex(RuntimeError, "no EE pose source available"):
            ex._resolve_ee_pose_or_fail(np.array([0.0, 0.0, 0.0]))

    def test_execute_with_safety_rejects_when_predicted_z_unsafe_even_if_current_z_safe(self):
        ex = object.__new__(GazeboRuntimeL3Executor)
        ex.q_min = np.array([-1.0] * 7, dtype=float)
        ex.q_max = np.array([1.0] * 7, dtype=float)
        ex._accepted_low_motion_streak = 0
        ex.feasibility_eps = 1e-6
        ex.null_effect_eps = 1e-4
        ex._normalize_cmd = lambda cmd, n: (True, np.asarray(cmd, dtype=float).copy(), ("L3_CHECK:ok",))
        ex.read_runtime_state = lambda n_joints: (
            np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.zeros(7, dtype=float),
            np.array([0.0, 0.0, 0.30], dtype=float),
        )
        ex._predict_ee_pose_from_q_target = lambda **_kwargs: (np.array([0.0, 0.0, 0.15], dtype=float), "unit_test_pred")

        state = Task1State(
            q=np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0], dtype=float),
            dq=np.zeros(7, dtype=float),
            target_pose_xyz=np.array([0.35, 0.0, 0.35], dtype=float),
            step=0,
            max_steps=40,
            safe_z_min=0.20,
            ee_proxy_xyz=np.array([0.0, 0.0, 0.30], dtype=float),
        )

        out = ex.execute_with_safety(state, np.array([0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0], dtype=float))

        self.assertFalse(out.accepted)
        self.assertTrue(any("predicted_ee_z=" in s for s in out.logs))
        self.assertTrue(any("current_ee_z=" in s for s in out.logs))
        self.assertTrue(any("safe_z_min=" in s for s in out.logs))
        self.assertTrue(any("reason=predicted_z_under_safe_min" in s for s in out.logs))

    def test_safe_z_min_parameter_effective(self):
        rows = run_task1_training(
            episodes=1,
            reward_mode="heuristic",
            backend="bootstrap",
            cfg=Task1Config(max_steps=1, safe_z_min=0.33),
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["safe_z_min"]), 0.33)

    def test_cli_n_joints_overrides_config(self):
        captured_cfg = {}

        def _fake_run_task1_training(*, cfg, **_kwargs):
            captured_cfg["cfg"] = cfg
            return [{"success": True, "total_reward": 0.0}]

        with mock.patch.object(task1_train, "run_task1_training", side_effect=_fake_run_task1_training):
            rc = task1_train.main(["--backend", "bootstrap", "--episodes", "1", "--n-joints", "7"])

        self.assertEqual(rc, 0)
        self.assertIn("cfg", captured_cfg)
        self.assertEqual(captured_cfg["cfg"].n_joints, 7)

    def test_cli_backend_default_n_joints_preserves_bootstrap_compat(self):
        captured_cfg = {}

        def _fake_run_task1_training(*, cfg, **_kwargs):
            captured_cfg["cfg"] = cfg
            return [{"success": True, "total_reward": 0.0}]

        with mock.patch.object(task1_train, "run_task1_training", side_effect=_fake_run_task1_training):
            rc = task1_train.main(["--backend", "bootstrap", "--episodes", "1"])

        self.assertEqual(rc, 0)
        self.assertIn("cfg", captured_cfg)
        self.assertEqual(captured_cfg["cfg"].n_joints, Task1Config.n_joints)

    def test_cli_backend_default_n_joints_uses_7_for_gazebo(self):
        captured_cfg = {}

        def _fake_run_task1_training(*, cfg, **_kwargs):
            captured_cfg["cfg"] = cfg
            return [{"success": True, "total_reward": 0.0}]

        with mock.patch.object(task1_train, "run_task1_training", side_effect=_fake_run_task1_training):
            rc = task1_train.main(["--backend", "gazebo", "--episodes", "1"])

        self.assertEqual(rc, 0)
        self.assertIn("cfg", captured_cfg)
        self.assertEqual(captured_cfg["cfg"].n_joints, 7)


if __name__ == "__main__":
    unittest.main()
