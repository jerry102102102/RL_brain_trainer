from __future__ import annotations

import unittest

import numpy as np

from hrl_trainer.v5_1.runtime_ros2 import JointStateFrame, RuntimeROS2Adapter


class _FakeIO:
    def __init__(self) -> None:
        self.published: list[dict[str, object]] = []
        self.frames = [
            JointStateFrame(names=["j1", "j2"], position=[0.0, 0.0], velocity=[0.0, 0.0], stamp_ns=1),
            JointStateFrame(names=["j1", "j2"], position=[0.1, -0.2], velocity=[0.0, 0.0], stamp_ns=2),
        ]

    def publish_joint_target(self, joint_names: list[str], positions: np.ndarray, duration_s: float) -> None:
        self.published.append(
            {
                "joint_names": list(joint_names),
                "positions": positions.tolist(),
                "duration_s": float(duration_s),
            }
        )

    def wait_for_joint_state(self, timeout_s: float) -> JointStateFrame:
        if not self.frames:
            raise TimeoutError("no frame")
        return self.frames.pop(0)


class TestRuntimeROS2Adapter(unittest.TestCase):
    def test_step_emits_cmd_and_readback(self) -> None:
        io = _FakeIO()
        adapter = RuntimeROS2Adapter(io=io, joint_names=["j1", "j2"], command_duration_s=0.3, settle_timeout_s=1.0)

        out = adapter.step(np.array([0.2, -0.1], dtype=float))

        self.assertEqual(len(io.published), 1)
        self.assertEqual(io.published[0]["joint_names"], ["j1", "j2"])
        self.assertEqual(out["q_before"], [0.0, 0.0])
        self.assertEqual(out["q_after"], [0.1, -0.2])
        self.assertAlmostEqual(float(out["joint_delta_l2"]), float(np.linalg.norm(np.array([0.1, -0.2]))), places=6)

    def test_read_q_requires_requested_joint_names(self) -> None:
        class _MissingJointIO(_FakeIO):
            def __init__(self) -> None:
                self.published = []
                self.frames = [JointStateFrame(names=["j1"], position=[0.0], velocity=[0.0], stamp_ns=1)]

        adapter = RuntimeROS2Adapter(io=_MissingJointIO(), joint_names=["j1", "j2"])
        with self.assertRaises(ValueError):
            _ = adapter.read_q()

    def test_step_waits_for_newer_joint_state_frame(self) -> None:
        class _StaleThenFreshIO(_FakeIO):
            def __init__(self) -> None:
                self.published = []
                stale = JointStateFrame(names=["j1", "j2"], position=[0.0, 0.0], velocity=[0.0, 0.0], stamp_ns=10)
                fresh = JointStateFrame(names=["j1", "j2"], position=[0.25, -0.1], velocity=[0.0, 0.0], stamp_ns=11)
                self.frames = [stale, stale, fresh]

        adapter = RuntimeROS2Adapter(io=_StaleThenFreshIO(), joint_names=["j1", "j2"], settle_timeout_s=0.3)
        out = adapter.step(np.array([0.25, -0.1], dtype=float))
        self.assertEqual(out["frame_before_stamp_ns"], 10)
        self.assertEqual(out["frame_after_stamp_ns"], 11)
        self.assertGreater(out["joint_delta_l2"], 1e-4)

    def test_step_rejects_command_shape_mismatch(self) -> None:
        io = _FakeIO()
        adapter = RuntimeROS2Adapter(io=io, joint_names=["j1", "j2"])
        with self.assertRaises(ValueError):
            _ = adapter.step(np.array([0.1], dtype=float))

    def test_step_skips_publish_for_tiny_command(self) -> None:
        io = _FakeIO()
        adapter = RuntimeROS2Adapter(io=io, joint_names=["j1", "j2"], min_command_l2=1e-3)
        out = adapter.step(np.array([1e-5, 1e-5], dtype=float))
        self.assertEqual(len(io.published), 0)
        self.assertTrue(bool(out["no_effect"]))
        self.assertEqual(out["no_effect_reason"], "below_min_command")
        self.assertTrue(bool(out["skipped_publish"]))

    def test_step_detects_no_effect_by_ratio(self) -> None:
        class _WeakEffectIO(_FakeIO):
            def __init__(self) -> None:
                self.published = []
                self.frames = [
                    JointStateFrame(names=["j1", "j2"], position=[0.0, 0.0], velocity=[0.0, 0.0], stamp_ns=20),
                    JointStateFrame(names=["j1", "j2"], position=[0.001, 0.0], velocity=[0.0, 0.0], stamp_ns=21),
                    JointStateFrame(names=["j1", "j2"], position=[0.001, 0.0], velocity=[0.0, 0.0], stamp_ns=22),
                    JointStateFrame(names=["j1", "j2"], position=[0.001, 0.0], velocity=[0.0, 0.0], stamp_ns=23),
                ]

        adapter = RuntimeROS2Adapter(
            io=_WeakEffectIO(),
            joint_names=["j1", "j2"],
            no_effect_l2=1e-6,
            no_effect_ratio=0.2,
            settle_hold_s=0.0,
        )
        out = adapter.step(np.array([0.5, 0.0], dtype=float))
        self.assertTrue(bool(out["no_effect"]))
        self.assertEqual(out["no_effect_reason"], "small_effect_ratio")


if __name__ == "__main__":
    unittest.main()
