from __future__ import annotations

import importlib.util
import json
import unittest

from hrl_trainer.v5_1 import pipeline_e2e
from hrl_trainer.v5_1.pipeline_e2e import run_pipeline_e2e

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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
                policy_mode="rule",
            )

            summary_path = tmp_path / "artifacts" / "pipeline_summary.json"
            curriculum_path = tmp_path / "artifacts" / "curriculum_state.json"
            gate_path = tmp_path / "artifacts" / "gate_result.json"

            self.assertEqual(out["summary"], str(summary_path))
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["exit_code"], 0)
            self.assertTrue(summary_path.exists())
            self.assertTrue((tmp_path / "artifacts" / "reward_trace.jsonl").exists())
            self.assertTrue(curriculum_path.exists())
            self.assertTrue(gate_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["episodes"]), 4)
            self.assertIn("metrics", summary)
            self.assertIn("artifacts", summary)
            self.assertEqual(summary["gate_overall_decision"], "GO")

            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate_payload["overall_decision"], "GO")

            logs_root = tmp_path / "artifacts" / "logs"
            self.assertTrue((logs_root / "l1").exists())
            self.assertTrue((logs_root / "l2").exists())
            self.assertTrue((logs_root / "l3").exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
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
                    policy_mode="rule",
                )

                self.assertEqual(out["status"], "gates_blocked")
                self.assertNotEqual(out["exit_code"], 0)

                gate_payload = json.loads((tmp_path / "artifacts" / "gate_result.json").read_text(encoding="utf-8"))
                self.assertEqual(gate_payload["overall_decision"], "HOLD")
        finally:
            pipeline_e2e.run_smoke = original


if __name__ == "__main__":
    unittest.main()
