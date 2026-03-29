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
            self.assertTrue((tmp_path / "artifacts" / "episode_reward_summary.jsonl").exists())
            self.assertTrue((tmp_path / "artifacts" / "runtime_trace.jsonl").exists())
            self.assertTrue(curriculum_path.exists())
            self.assertTrue(gate_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["episodes"]), 4)
            self.assertIn("metrics", summary)
            self.assertIn("artifacts", summary)
            self.assertEqual(summary["gate_overall_decision"], "GO")
            self.assertEqual(summary["policy_mode"], "sac_torch")
            self.assertEqual(summary["stage_profile"], "default")

            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 1)
            self.assertIn("episode_id", reward_rows[0])
            self.assertIn("reward_total", reward_rows[0])
            self.assertEqual(
                set(reward_rows[0]["components"].keys()),
                {
                    "progress",
                    "action",
                    "jerk",
                    "intervention",
                    "clamp_or_projection",
                    "stall",
                    "timeout_or_reset",
                    "success_bonus",
                    "reward_total",
                },
            )

            ep_summary_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts" / "episode_reward_summary.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertEqual(len(ep_summary_rows), 4)
            self.assertIn("component_sums", ep_summary_rows[0])
            self.assertIn("component_means", ep_summary_rows[0])
            self.assertIn("total_reward", ep_summary_rows[0])

            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate_payload["overall_decision"], "GO")

            logs_root = tmp_path / "artifacts" / "logs"
            self.assertTrue((logs_root / "l1").exists())
            self.assertTrue((logs_root / "l2").exists())
            self.assertTrue((logs_root / "l3").exists())

    def test_pipeline_e2e_supports_s0_b_stage_profile(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_s0b",
                episodes=1,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_s0b",
                stage_profile="s0_b",
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_s0b" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stage_profile"], "s0_b")
            self.assertEqual(summary["episodes"][0]["stage"], "S0_B")

            l2_path = Path(summary["episodes"][0]["logs"]["l2"])
            l2_rows = [json.loads(x) for x in l2_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertGreaterEqual(len(l2_rows), 1)
            for row in l2_rows:
                clipped = row["payload"]["action_clipped"]
                self.assertTrue(all(abs(float(v)) <= 0.15 + 1e-9 for v in clipped))

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

    def test_pipeline_e2e_gz_mode_supports_rack_joint_passthrough(self) -> None:
        from pathlib import Path
        import tempfile

        class _FakeRuntime7:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                run_id="test_e2e_gz_rack_passthrough",
                episodes=1,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_gz_rack",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _FakeRuntime7(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            runtime_trace = tmp_path / "artifacts_gz_rack" / "runtime_trace.jsonl"
            rows = [json.loads(x) for x in runtime_trace.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["cmd_q"]), 7)
            self.assertAlmostEqual(float(rows[0]["cmd_q"][0]), 0.0, places=7)

    def test_pipeline_e2e_gz_mode_failfast_no_effect(self) -> None:
        from pathlib import Path
        import tempfile

        class _NoEffectRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6

            def read_q(self, timeout_s=None):
                import numpy as _np

                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                import numpy as _np

                before = _np.asarray(self._q, dtype=float)
                # no movement regardless of command
                after = before.copy()
                return {
                    "q_before": before.tolist(),
                    "q_after": after.tolist(),
                    "cmd_q": _np.asarray(cmd_q, dtype=float).tolist(),
                    "joint_delta": (after - before).tolist(),
                    "joint_delta_l2": 0.0,
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_gz_no_effect",
                episodes=1,
                steps_per_episode=6,
                artifact_root=tmp_path / "artifacts_gz_no_effect",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _NoEffectRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts_gz_no_effect" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 3)
            self.assertEqual(reward_rows[-1]["done_reason"], "no_effect")

    def test_pipeline_e2e_reward_allows_normal_path_on_execution_success(self) -> None:
        from pathlib import Path
        import tempfile

        class _ExecSuccessRuntime:
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
                    "accepted": True,
                    "result_status": "success",
                    "execution_ok": True,
                    "fail_reason": "none",
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_exec_success",
                episodes=1,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_exec_success",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _ExecSuccessRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts_exec_success" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 1)
            self.assertTrue(all(row["done_reason"] != "execution_fail" for row in reward_rows))
            self.assertTrue(any(abs(float(row["components"]["progress"])) > 0.0 for row in reward_rows))

    def test_pipeline_e2e_reward_uses_fail_penalty_on_execution_fail(self) -> None:
        from pathlib import Path
        import tempfile

        class _ExecFailRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6

            def read_q(self, timeout_s=None):
                import numpy as _np

                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                import numpy as _np

                before = _np.asarray(self._q, dtype=float)
                after = before.copy()
                return {
                    "q_before": before.tolist(),
                    "q_after": after.tolist(),
                    "cmd_q": _np.asarray(cmd_q, dtype=float).tolist(),
                    "joint_delta": (after - before).tolist(),
                    "joint_delta_l2": 0.0,
                    "accepted": True,
                    "result_status": "fail",
                    "execution_ok": False,
                    "fail_reason": "controller_rejected",
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_exec_fail",
                episodes=1,
                steps_per_episode=4,
                artifact_root=tmp_path / "artifacts_exec_fail",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _ExecFailRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts_exec_fail" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 1)
            fail_row = reward_rows[-1]
            self.assertEqual(fail_row["done_reason"], "execution_fail")
            self.assertEqual(float(fail_row["components"]["progress"]), 0.0)
            self.assertEqual(float(fail_row["components"]["action"]), 0.0)
            self.assertEqual(float(fail_row["components"]["jerk"]), 0.0)
            self.assertLess(float(fail_row["components"]["timeout_or_reset"]), 0.0)

    def test_pipeline_e2e_reward_uses_fail_penalty_on_action_rejected(self) -> None:
        from pathlib import Path
        import tempfile

        class _RejectedRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6

            def read_q(self, timeout_s=None):
                import numpy as _np

                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                import numpy as _np

                before = _np.asarray(self._q, dtype=float)
                after = before.copy()
                return {
                    "q_before": before.tolist(),
                    "q_after": after.tolist(),
                    "cmd_q": _np.asarray(cmd_q, dtype=float).tolist(),
                    "joint_delta": (after - before).tolist(),
                    "joint_delta_l2": 0.0,
                    "accepted": False,
                    "result_status": "rejected",
                    "execution_ok": False,
                    "fail_reason": "goal_rejected",
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_exec_rejected",
                episodes=1,
                steps_per_episode=4,
                artifact_root=tmp_path / "artifacts_exec_rejected",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _RejectedRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts_exec_rejected" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 1)
            fail_row = reward_rows[-1]
            self.assertEqual(fail_row["done_reason"], "execution_fail")
            self.assertEqual(float(fail_row["components"]["progress"]), 0.0)
            self.assertLess(float(fail_row["components"]["timeout_or_reset"]), 0.0)

    def test_pipeline_e2e_gz_mode_requires_joint_names(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline_e2e(
                run_id="test_e2e_gz_missing_names",
                episodes=1,
                steps_per_episode=1,
                artifact_root="/tmp/unused",
                runtime_mode="gz",
            )

    def test_pipeline_e2e_gz_mode_rejects_non_6_controlled_dofs(self) -> None:
        with self.assertRaises(ValueError):
            run_pipeline_e2e(
                run_id="test_e2e_gz_bad_joint_dim",
                episodes=1,
                steps_per_episode=1,
                artifact_root="/tmp/unused",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5"],
                runtime_factory=lambda **kwargs: object(),
            )

    def test_pipeline_e2e_reset_fail_aborts_episode(self) -> None:
        from pathlib import Path
        import tempfile

        class _ResetFailRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6
                self.calls = 0

            def read_q(self, timeout_s=None):
                import numpy as _np
                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                self.calls += 1
                return {
                    "q_before": list(self._q), "q_after": list(self._q), "cmd_q": list(cmd_q),
                    "joint_delta": [0.0] * 6, "joint_delta_l2": 0.0, "timestamp_ns": 123,
                    "accepted": False, "result_status": "rejected", "execution_ok": False, "fail_reason": "goal_rejected",
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_reset_fail",
                episodes=2,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_reset_fail",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _ResetFailRuntime(**kwargs),
            )
            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_reset_fail" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["reset_failures"], 1)
            self.assertEqual(summary["metrics"]["episodes_completed"], 0)

    def test_pipeline_e2e_reset_success_recorded(self) -> None:
        from pathlib import Path
        import tempfile

        class _OkRuntime:
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
                return {"q_before": before.tolist(), "q_after": after.tolist(), "cmd_q": after.tolist(), "joint_delta": (after - before).tolist(), "joint_delta_l2": float(_np.linalg.norm(after-before)), "timestamp_ns": 123, "accepted": True, "result_status": "success", "execution_ok": True, "fail_reason": "none"}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            run_pipeline_e2e(
                run_id="test_e2e_reset_ok",
                episodes=1,
                steps_per_episode=1,
                artifact_root=tmp_path / "artifacts_reset_ok",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _OkRuntime(**kwargs),
            )
            summary = json.loads((tmp_path / "artifacts_reset_ok" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes"][0]["reset_result"]["result_status"], "success")

    def test_pipeline_e2e_near_home_reset_skip_counts_as_success(self) -> None:
        from pathlib import Path
        import tempfile

        class _NearHomeRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 6
                self.step_calls = 0

            def read_q(self, timeout_s=None):
                import numpy as _np
                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                import numpy as _np
                self.step_calls += 1
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
                    "accepted": True,
                    "result_status": "success",
                    "execution_ok": True,
                    "fail_reason": "none",
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            runtime_holder: dict[str, _NearHomeRuntime] = {}

            def _factory(**kwargs):
                runtime_holder["rt"] = _NearHomeRuntime(**kwargs)
                return runtime_holder["rt"]

            run_pipeline_e2e(
                run_id="test_e2e_near_home_skip",
                episodes=1,
                steps_per_episode=1,
                artifact_root=tmp_path / "artifacts_near_home_skip",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=_factory,
                reset_near_home_eps=1e-3,
            )

            summary = json.loads((tmp_path / "artifacts_near_home_skip" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["reset_failures"], 0)
            self.assertTrue(bool(summary["episodes"][0]["reset_result"].get("reset_skipped_near_home", False)))

            ep_rows = [json.loads(x) for x in (tmp_path / "artifacts_near_home_skip" / "episode_reward_summary.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(bool(ep_rows[0].get("reset_skipped_near_home", False)))

    def test_pipeline_e2e_runtime_trace_includes_ee_fields(self) -> None:
        from pathlib import Path
        import tempfile

        class _OkRuntime:
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
                return {"q_before": before.tolist(), "q_after": after.tolist(), "cmd_q": after.tolist(), "joint_delta": (after - before).tolist(), "joint_delta_l2": float(_np.linalg.norm(after-before)), "timestamp_ns": 123, "accepted": True, "result_status": "success", "execution_ok": True, "fail_reason": "none"}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            run_pipeline_e2e(
                run_id="test_e2e_ee_trace",
                episodes=1,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_ee_trace",
                runtime_mode="gz",
                runtime_joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _OkRuntime(**kwargs),
            )
            rows = [json.loads(x) for x in (tmp_path / "artifacts_ee_trace" / "runtime_trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            step_rows = [r for r in rows if r.get("step", 0) >= 0]
            self.assertIn("ee_pose", step_rows[0])
            self.assertIn("ee_target", step_rows[0])
            self.assertIn("ee_pos_err", step_rows[0])
            self.assertIn("ee_ori_err", step_rows[0])


if __name__ == "__main__":
    unittest.main()
