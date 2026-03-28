from __future__ import annotations

import importlib.util
import json
import unittest

from hrl_trainer.v5_1 import pipeline_e2e
from hrl_trainer.v5_1.pipeline_e2e import run_pipeline_e2e

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class TestV51PipelineE2E(unittest.TestCase):
    def test_pipeline_e2e_outputs_artifacts_and_logs(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e",
                episodes=4,
                steps_per_episode=3,
                artifact_root=tmp_path / "artifacts",
            )

            summary_path = tmp_path / "artifacts" / "pipeline_summary.json"
            curriculum_path = tmp_path / "artifacts" / "curriculum_state.json"
            gate_path = tmp_path / "artifacts" / "gate_result.json"

            self.assertEqual(out["summary"], str(summary_path))
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["exit_code"], 0)
            self.assertTrue(summary_path.exists())
            self.assertTrue((tmp_path / "artifacts" / "reward_trace.jsonl").exists())
            self.assertTrue((tmp_path / "artifacts" / "runtime_trace.jsonl").exists())
            self.assertTrue(curriculum_path.exists())
            self.assertTrue(gate_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["episodes"]), 4)
            self.assertIn("metrics", summary)
            self.assertIn("artifacts", summary)
            self.assertEqual(summary["gate_overall_decision"], "GO")
            self.assertEqual(summary["policy_mode"], "sac_torch")

            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate_payload["overall_decision"], "GO")

            logs_root = tmp_path / "artifacts" / "logs"
            self.assertTrue((logs_root / "l1").exists())
            self.assertTrue((logs_root / "l2").exists())
            self.assertTrue((logs_root / "l3").exists())

    def test_pipeline_e2e_supports_sac_torch_policy_mode(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_sac",
                episodes=3,
                steps_per_episode=12,
                artifact_root=tmp_path / "artifacts_sac",
                policy_mode="sac_torch",
                sac_seed=11,
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_sac" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["policy_mode"], "sac_torch")
            self.assertIn("train_metrics", summary)
            self.assertGreaterEqual(len(summary["train_metrics"]), 1)

    def test_pipeline_e2e_rejects_non_torch_policy_mode(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline_e2e(
                run_id="test_e2e_bad_mode",
                episodes=1,
                steps_per_episode=2,
                artifact_root="/tmp/unused",
                policy_mode="rule",
            )

    def test_pipeline_e2e_enforce_gates_returns_nonzero_on_fail(self) -> None:
        from pathlib import Path
        import tempfile

        def _boom(*args, **kwargs):
            raise RuntimeError("reset failed")

        original = pipeline_e2e.run_smoke
        pipeline_e2e.run_smoke = _boom
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td)
                out = run_pipeline_e2e(
                    run_id="test_e2e_fail",
                    episodes=2,
                    steps_per_episode=3,
                    artifact_root=tmp_path / "artifacts",
                    enforce_gates=True,
                )

                self.assertEqual(out["status"], "gates_blocked")
                self.assertNotEqual(out["exit_code"], 0)

                gate_payload = json.loads((tmp_path / "artifacts" / "gate_result.json").read_text(encoding="utf-8"))
                self.assertEqual(gate_payload["overall_decision"], "HOLD")
        finally:
            pipeline_e2e.run_smoke = original

    def test_pipeline_e2e_gz_mode_writes_runtime_trace(self) -> None:
        from pathlib import Path
        import tempfile

        class _FakeRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6

            def read_q(self, timeout_s=None):
                import numpy as _np

                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                import numpy as _np

                before = _np.asarray(self._q, dtype=float)
                after = _np.asarray(cmd_q, dtype=float)
                self._q = after.tolist()
                return {
                    "q_before": before.tolist(),
                    "q_after": after.tolist(),
                    "cmd_q": after.tolist(),
                    "joint_delta": (after - before).tolist(),
                    "joint_delta_l2": float(_np.linalg.norm(after - before)),
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_gz",
                episodes=2,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_gz",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _FakeRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            runtime_trace = tmp_path / "artifacts_gz" / "runtime_trace.jsonl"
            self.assertTrue(runtime_trace.exists())
            self.assertGreater(len(runtime_trace.read_text(encoding="utf-8").strip().splitlines()), 0)

            summary = json.loads((tmp_path / "artifacts_gz" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_mode"], "gz")
            self.assertGreaterEqual(len(summary["episode_joint_delta_summary"]), 1)

    def test_pipeline_e2e_gz_mode_requires_joint_names(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline_e2e(
                run_id="test_e2e_gz_missing_names",
                episodes=1,
                steps_per_episode=1,
                artifact_root="/tmp/unused",
                runtime_mode="gz",
            )


if __name__ == "__main__":
    unittest.main()
