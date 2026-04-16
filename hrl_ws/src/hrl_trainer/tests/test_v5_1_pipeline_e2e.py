from __future__ import annotations

import importlib.util
import json
import unittest

import numpy as np

from hrl_trainer.v5_1 import pipeline_e2e
from hrl_trainer.v5_1.pipeline_e2e import run_pipeline_e2e

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestV51PipelineE2EScoring(unittest.TestCase):
    def test_checkpoint_score_prioritizes_deterministic_quality(self) -> None:
        stronger_deterministic = {
            "det_success_rate": 0.2,
            "mean_final_dpos": 0.12,
            "regression_rate": 0.60,
            "mean_final_minus_min": 0.02,
        }
        weaker_deterministic = {
            "det_success_rate": 0.0,
            "mean_final_dpos": 0.10,
            "regression_rate": 0.55,
            "mean_final_minus_min": 0.01,
        }

        self.assertGreater(
            pipeline_e2e._checkpoint_score(stronger_deterministic),
            pipeline_e2e._checkpoint_score(weaker_deterministic),
        )

    def test_fixed_eval_suite_is_deterministic(self) -> None:
        suite_a = pipeline_e2e._build_fixed_eval_suite(
            suite_size=3,
            suite_seed=123,
            target_mode="near_home",
            action_stage_name="S2",
            target_curriculum_stage_name="TC0",
            near_home_profile="TC0",
            near_home_pos_offset_min_m=0.08,
            near_home_pos_offset_max_m=0.10,
            near_home_ori_offset_min_deg=0.0,
            near_home_ori_offset_max_deg=2.0,
            external_ee_target=np.zeros(6, dtype=float),
            external_ee_target_source={},
        )
        suite_b = pipeline_e2e._build_fixed_eval_suite(
            suite_size=3,
            suite_seed=123,
            target_mode="near_home",
            action_stage_name="S2",
            target_curriculum_stage_name="TC0",
            near_home_profile="TC0",
            near_home_pos_offset_min_m=0.08,
            near_home_pos_offset_max_m=0.10,
            near_home_ori_offset_min_deg=0.0,
            near_home_ori_offset_max_deg=2.0,
            external_ee_target=np.zeros(6, dtype=float),
            external_ee_target_source={},
        )
        self.assertEqual(suite_a["suite_id"], suite_b["suite_id"])
        self.assertEqual(suite_a["targets"], suite_b["targets"])

    def test_exploration_schedule_reduces_after_good_progress(self) -> None:
        new_scale, reason = pipeline_e2e._schedule_exploration_scale(
            0.60,
            total_successes=5,
            best_min_dpos=0.05,
            det_success_rate=0.0,
        )
        self.assertEqual(new_scale, 0.45)
        self.assertEqual(reason, "train_success>=5")

    def test_disable_exploration_schedule_keeps_scale_fixed(self) -> None:
        new_scale, reason = pipeline_e2e._maybe_schedule_exploration_scale(
            True,
            0.60,
            total_successes=99,
            best_min_dpos=0.0,
            det_success_rate=1.0,
        )
        self.assertEqual(new_scale, 0.60)
        self.assertIsNone(reason)


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
            self.assertTrue(
                {
                    "progress",
                    "action",
                    "jerk",
                    "adjust_penalty",
                    "raw_action_penalty",
                    "reject_penalty",
                    "intervention",
                    "clamp_or_projection",
                    "near_goal",
                    "near_goal_shell",
                    "dwell",
                    "near_goal_exit",
                    "local_drift_penalty",
                    "dwell_break",
                    "timeout_or_reset",
                    "success_bonus",
                    "success_triggered_by_dwell",
                    "reward_total",
                }.issubset(set(reward_rows[0]["components"].keys()))
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
            self.assertIn("max_dwell_count", ep_summary_rows[0])
            self.assertIn("dwell_break_count", ep_summary_rows[0])
            self.assertIn("clamp_count", ep_summary_rows[0])
            self.assertIn("projection_count", ep_summary_rows[0])
            self.assertIn("reject_count", ep_summary_rows[0])
            self.assertIn("reject_rate", ep_summary_rows[0])
            self.assertIn("near_goal_entry_count", ep_summary_rows[0])
            self.assertIn("near_goal_shell_count", ep_summary_rows[0])
            self.assertIn("near_goal_exit_count", ep_summary_rows[0])
            self.assertIn("success_triggered_by_dwell", ep_summary_rows[0])
            self.assertIn("sum_adjust_penalty", ep_summary_rows[0])
            self.assertIn("sum_raw_action_penalty", ep_summary_rows[0])
            self.assertIn("sum_reject_penalty", ep_summary_rows[0])
            self.assertIn("sum_delta_norm", ep_summary_rows[0])
            self.assertIn("min_dpos", ep_summary_rows[0])
            self.assertIn("final_dpos", ep_summary_rows[0])
            self.assertIn("final_dpos_minus_min_dpos", ep_summary_rows[0])

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

    def test_near_home_target_generation_logic(self) -> None:
        ee_target, source = pipeline_e2e._resolve_near_home_ee_target(
            home_q=np.zeros(7, dtype=float),
            profile="s0_bootstrap",
            pos_offset_min_m=0.02,
            pos_offset_max_m=0.05,
            ori_offset_min_deg=5.0,
            ori_offset_max_deg=10.0,
            rng=np.random.default_rng(7),
        )

        delta_pos = np.asarray(source["target_delta_pos"], dtype=float)
        delta_ori = np.asarray(source["target_delta_ori"], dtype=float)
        self.assertGreaterEqual(float(np.linalg.norm(delta_pos)), 0.02 - 1e-9)
        self.assertLessEqual(float(np.linalg.norm(delta_pos)), 0.05 + 1e-9)
        self.assertGreaterEqual(float(np.linalg.norm(delta_ori)), np.deg2rad(5.0) - 1e-9)
        self.assertLessEqual(float(np.linalg.norm(delta_ori)), np.deg2rad(10.0) + 1e-9)
        self.assertTrue(np.allclose(np.asarray(source["home_ee"], dtype=float) + np.concatenate([delta_pos, delta_ori]), ee_target, atol=1e-6))
        self.assertLessEqual(float(ee_target[2]), float(source["home_ee"][2]) + 1e-9)
        self.assertLessEqual(float(delta_pos[2]), 1e-9)
        self.assertTrue(bool(source["z_not_above_home"]))

    def test_s0_b_default_target_mode_is_near_home(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_s0b_near_home",
                episodes=1,
                steps_per_episode=1,
                artifact_root=tmp_path / "artifacts_s0b_near_home",
                stage_profile="s0_b",
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_s0b_near_home" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["target_mode"], "auto")
            self.assertEqual(summary["resolved_target_mode"], "near_home")
            self.assertEqual(summary["episodes"][0]["target_mode"], "near_home")
            self.assertEqual(summary["ee_target_source"]["provider"], "near_home_randomized")
            self.assertIn("home_ee", summary["ee_target_source"])
            self.assertIn("target_delta_pos", summary["ee_target_source"])
            self.assertIn("target_delta_ori", summary["ee_target_source"])
            self.assertIn("home_ee", summary["episodes"][0])
            self.assertIn("target_delta_pos", summary["episodes"][0])
            self.assertIn("target_delta_ori", summary["episodes"][0])

    def test_near_home_target_changes_across_episodes(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_s0b_near_home_randomized",
                episodes=2,
                steps_per_episode=1,
                artifact_root=tmp_path / "artifacts_s0b_near_home_randomized",
                stage_profile="s0_b",
                sac_seed=5,
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_s0b_near_home_randomized" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes"][0]["target_mode"], "near_home")
            target0 = np.asarray(summary["episodes"][0]["ee_target"], dtype=float)
            target1 = np.asarray(summary["episodes"][1]["ee_target"], dtype=float)
            self.assertFalse(np.allclose(target0, target1, atol=1e-9))

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

    def test_pipeline_e2e_learning_dynamics_param_hash_changes_with_training(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out = run_pipeline_e2e(
                run_id="test_e2e_learning_dynamics",
                episodes=6,
                steps_per_episode=20,
                artifact_root=tmp_path / "artifacts_learning_dynamics",
                policy_mode="sac_torch",
                sac_seed=7,
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_learning_dynamics" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertIn("train_metrics", summary)
            self.assertGreaterEqual(len(summary["train_metrics"]), 1)

            last = summary["train_metrics"][-1]
            self.assertGreater(float(last["updates_applied"]), 0.0)
            self.assertIn("param_hash_actor", last)
            self.assertIn("param_hash_critic", last)

            actor_hashes = [m["param_hash_actor"] for m in summary["train_metrics"] if m.get("param_hash_actor")]
            critic_hashes = [m["param_hash_critic"] for m in summary["train_metrics"] if m.get("param_hash_critic")]
            self.assertGreaterEqual(len(set(actor_hashes)), 2)
            self.assertGreaterEqual(len(set(critic_hashes)), 2)
            self.assertTrue(bool(summary.get("learning_effective", False)))
            self.assertEqual(summary.get("ineffective_reasons", []), [])

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
                run_id="test_e2e_gz",
                episodes=2,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_gz",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _FakeRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            runtime_trace = tmp_path / "artifacts_gz" / "runtime_trace.jsonl"
            self.assertTrue(runtime_trace.exists())
            self.assertGreater(len(runtime_trace.read_text(encoding="utf-8").strip().splitlines()), 0)

            summary = json.loads((tmp_path / "artifacts_gz" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_mode"], "gz")
            self.assertGreaterEqual(len(summary["episode_joint_delta_summary"]), 1)

    def test_pipeline_e2e_gz_mode_records_target_visualization_status(self) -> None:
        from pathlib import Path
        import tempfile

        class _FakeRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7
                self.visual_calls: list[list[float]] = []

            def read_q(self, timeout_s=None):
                import numpy as _np
                return _np.asarray(self._q, dtype=float)

            def publish_ee_target_visual(self, ee_target):
                import numpy as _np
                self.visual_calls.append(_np.asarray(ee_target, dtype=float).tolist())
                return {
                    "success": True,
                    "action": "create",
                    "reason": "ok",
                    "world_name": "empty",
                    "entity_name": "unit_target",
                }

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
                    "accepted": True,
                    "result_status": "success",
                    "execution_ok": True,
                    "fail_reason": "none",
                }

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            runtime_holder: dict[str, _FakeRuntime] = {}

            def _factory(**kwargs):
                runtime_holder["rt"] = _FakeRuntime(**kwargs)
                return runtime_holder["rt"]

            out = run_pipeline_e2e(
                run_id="test_e2e_gz_target_visual",
                episodes=1,
                steps_per_episode=2,
                artifact_root=tmp_path / "artifacts_gz_target_visual",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=_factory,
            )

            self.assertEqual(out["exit_code"], 0)
            self.assertEqual(len(runtime_holder["rt"].visual_calls), 1)
            summary = json.loads((tmp_path / "artifacts_gz_target_visual" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertIn("target_visualization", summary["episodes"][0])
            self.assertTrue(bool(summary["episodes"][0]["target_visualization"]["success"]))
            self.assertTrue(bool(summary["gz_target_visualization_enabled"]))

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
            step_rows = [r for r in rows if "cmd_q" in r]
            self.assertGreaterEqual(len(step_rows), 1)
            self.assertEqual(len(step_rows[0]["cmd_q"]), 7)

    def test_controlled_indices_and_expand_include_rack_joint(self) -> None:
        indices = pipeline_e2e._controlled_joint_indices(["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"])
        self.assertEqual(indices, [0, 1, 2, 3, 4, 5, 6])

        q_before = np.array([0.11, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=float)
        q_des = np.array([-0.22, -0.21, 0.31, -0.41, 0.51, -0.61, 0.71], dtype=float)
        q_cmd = pipeline_e2e._expand_cmd_q(q_before, indices, q_des)
        self.assertAlmostEqual(float(q_cmd[0]), float(q_des[0]), places=9)
        np.testing.assert_allclose(q_cmd, q_des, atol=1e-12)

    def test_pipeline_e2e_gz_mode_failfast_no_effect(self) -> None:
        from pathlib import Path
        import tempfile

        class _NoEffectRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
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

    def test_pipeline_e2e_stops_early_on_success_by_dwell(self) -> None:
        from pathlib import Path
        import tempfile

        class _StaticHomeRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                    "result_status": "success",
                    "execution_ok": True,
                    "fail_reason": "none",
                    "timestamp_ns": 123,
                }

            def close(self):
                return None

        original_target_resolver = pipeline_e2e._resolve_near_home_ee_target

        def _home_target(*, home_q, **kwargs):
            home_ee = pipeline_e2e._ee_pose_from_q(np.asarray(home_q, dtype=float))
            return home_ee.copy(), {
                "provider": "near_home_randomized",
                "profile": "test",
                "home_q": np.asarray(home_q, dtype=float).tolist(),
                "home_ee": home_ee.tolist(),
                "target_delta_pos": [0.0, 0.0, 0.0],
                "target_delta_ori": [0.0, 0.0, 0.0],
            }

        pipeline_e2e._resolve_near_home_ee_target = _home_target
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td)
                out = run_pipeline_e2e(
                    run_id="test_e2e_success_by_dwell",
                    episodes=1,
                    steps_per_episode=10,
                    artifact_root=tmp_path / "artifacts_success_by_dwell",
                    runtime_mode="gz",
                    stage_profile="s0_b",
                    target_mode="near_home",
                    runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                    runtime_factory=lambda **kwargs: _StaticHomeRuntime(**kwargs),
                )

                self.assertEqual(out["exit_code"], 0)
                summary = json.loads((tmp_path / "artifacts_success_by_dwell" / "pipeline_summary.json").read_text(encoding="utf-8"))
                episode = summary["episodes"][0]
                self.assertEqual(episode["done_reason"], "success")
                self.assertTrue(bool(episode["success_triggered_by_dwell"]))
                self.assertEqual(float(episode["success_rate"]), 1.0)

                reward_rows = [
                    json.loads(x)
                    for x in (tmp_path / "artifacts_success_by_dwell" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                    if x.strip()
                ]
                self.assertEqual(len(reward_rows), 3)
                self.assertEqual(reward_rows[-1]["done_reason"], "success")
        finally:
            pipeline_e2e._resolve_near_home_ee_target = original_target_resolver

    def test_pipeline_e2e_reward_allows_normal_path_on_execution_success(self) -> None:
        from pathlib import Path
        import tempfile

        class _ExecSuccessRuntime:
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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
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
                self._q = [0.0] * 7

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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
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
            self.assertEqual(float(fail_row["components"]["ee_small_motion_penalty"]), 0.0)
            self.assertLess(float(fail_row["components"]["timeout_or_reset"]), 0.0)

    def test_pipeline_e2e_reward_trace_records_reward_state_transitions(self) -> None:
        from pathlib import Path
        import tempfile

        class _NoMotionButSuccessRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                run_id="test_e2e_reward_state_trace",
                episodes=1,
                steps_per_episode=1,
                artifact_root=tmp_path / "artifacts_reward_state_trace",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _NoMotionButSuccessRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            reward_rows = [
                json.loads(x)
                for x in (tmp_path / "artifacts_reward_state_trace" / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertGreaterEqual(len(reward_rows), 1)
            row = reward_rows[-1]
            self.assertIn("reward_state_in", row)
            self.assertIn("reward_state_out", row)
            self.assertIn("prev_in_dwell", row["reward_state_in"])
            self.assertIn("success_awarded", row["reward_state_out"])
            self.assertIn("dwell_break", row["components"])

    def test_pipeline_e2e_episode_summary_includes_new_reward_ablation_metrics(self) -> None:
        from pathlib import Path
        import tempfile

        class _StaticRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                run_id="test_e2e_reward_summary_metrics",
                episodes=1,
                steps_per_episode=3,
                artifact_root=tmp_path / "artifacts_reward_summary_metrics",
                runtime_mode="gz",
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _StaticRuntime(**kwargs),
            )

            self.assertEqual(out["exit_code"], 0)
            summary = json.loads((tmp_path / "artifacts_reward_summary_metrics" / "pipeline_summary.json").read_text(encoding="utf-8"))
            episode = summary["episodes"][0]
            self.assertIn("max_dwell_count", episode)
            self.assertIn("dwell_break_count", episode)
            self.assertIn("clamp_count", episode)
            self.assertIn("near_goal_entry_count", episode)
            self.assertIn("success_triggered_by_dwell", episode)
            self.assertIn("min_dpos", episode)
            self.assertIn("final_dpos", episode)

    def test_pipeline_e2e_reward_uses_fail_penalty_on_action_rejected(self) -> None:
        from pathlib import Path
        import tempfile

        class _RejectedRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7

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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
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

    def test_pipeline_e2e_gz_mode_rejects_non_7_controlled_dofs(self) -> None:
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
                self._q = [0.0] * 7
                self.calls = 0

            def read_q(self, timeout_s=None):
                import numpy as _np
                return _np.asarray(self._q, dtype=float)

            def step(self, cmd_q):
                self.calls += 1
                return {
                    "q_before": list(self._q), "q_after": list(self._q), "cmd_q": list(cmd_q),
                    "joint_delta": [0.0] * 7, "joint_delta_l2": 0.0, "timestamp_ns": 123,
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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _ResetFailRuntime(**kwargs),
                reset_near_home_eps=0.0,
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
                self._q = [0.0] * 7

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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _OkRuntime(**kwargs),
            )
            summary = json.loads((tmp_path / "artifacts_reset_ok" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes"][0]["reset_result"]["result_status"], "success")

    def test_pipeline_e2e_near_home_reset_skip_counts_as_success(self) -> None:
        from pathlib import Path
        import tempfile

        class _NearHomeRuntime:
            def __init__(self, **kwargs):
                self._q = [0.0] * 7
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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
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
                self._q = [0.0] * 7

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
                runtime_joint_names=["Rack_joint", "j1", "j2", "j3", "j4", "j5", "j6"],
                runtime_factory=lambda **kwargs: _OkRuntime(**kwargs),
            )
            rows = [json.loads(x) for x in (tmp_path / "artifacts_ee_trace" / "runtime_trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            step_rows = [r for r in rows if r.get("step", 0) >= 0]
            self.assertIn("ee_pose", step_rows[0])
            self.assertIn("ee_target", step_rows[0])
            self.assertIn("ee_pos_err", step_rows[0])
            self.assertIn("ee_ori_err", step_rows[0])

            summary = json.loads((tmp_path / "artifacts_ee_trace" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["resolved_target_mode"], "external")
            self.assertEqual(summary["ee_target_source"]["provider"], "external_task_library.MoveTaskLibrary.move_from_to")
            self.assertTrue(summary["ee_target_source"]["external_file"].endswith("kitchen_robot_controller/task_library.py"))

    def test_pipeline_e2e_auto_resume_and_checkpoint_persistence(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            artifact_root = tmp_path / "artifacts_resume"

            first = run_pipeline_e2e(
                run_id="test_e2e_resume_round1",
                episodes=2,
                steps_per_episode=8,
                artifact_root=artifact_root,
                policy_mode="sac_torch",
                sac_seed=13,
            )
            self.assertEqual(first["exit_code"], 0)
            first_summary = json.loads((artifact_root / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(first_summary["loaded_from_checkpoint"]))
            self.assertIsNone(first_summary["loaded_checkpoint_path"])
            self.assertEqual(first_summary["model_persistence_mode"], "auto-resume-required")
            self.assertGreaterEqual(len(first_summary["saved_checkpoint_paths"]), 1)
            for p in first_summary["saved_checkpoint_paths"]:
                self.assertTrue(Path(p).exists())

            second = run_pipeline_e2e(
                run_id="test_e2e_resume_round2",
                episodes=2,
                steps_per_episode=8,
                artifact_root=artifact_root,
                policy_mode="sac_torch",
                sac_seed=13,
            )
            self.assertEqual(second["exit_code"], 0)
            second_summary = json.loads((artifact_root / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(second_summary["loaded_from_checkpoint"]))
            self.assertIsNotNone(second_summary["loaded_checkpoint_path"])
            self.assertTrue(Path(second_summary["loaded_checkpoint_path"]).exists())
            self.assertEqual(second_summary["model_persistence_mode"], "auto-resume-required")
            self.assertGreaterEqual(len(second_summary["saved_checkpoint_paths"]), 1)
            for p in second_summary["saved_checkpoint_paths"]:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
