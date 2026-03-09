import unittest
from pathlib import Path

from hrl_trainer.v5.tools.wp0_healthcheck import (
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_PASS,
    ToolRunResult,
    attach_runner_metadata,
    build_report_skeleton,
    evaluate_approx_sync,
    evaluate_rosbag_replay,
    evaluate_tf_contract,
    evaluate_tray_stability,
    finalize_report,
)


class TestV5Wp0HealthcheckReport(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "wp0": {
                "use_sim_time": True,
                "window_sec": 60.0,
                "thresholds": {
                    "approx_sync_slop_ms": 50.0,
                    "approx_sync_success_rate_min": 0.95,
                    "pose_jitter_std_m": 0.003,
                    "id_switch_rate_max": 0.01,
                    "id_missing_warn_rate": 0.05,
                    "state_latency_p95_ms": 80.0,
                    "replay_image_latency_p95_ms": 120.0,
                },
                "approx_sync": {"queue_size": 10},
                "tf_checks": {"required_pairs": [{"source": "world", "target": "cam"}]},
                "rosbag": {"default_bag_path": "/tmp/test_bag"},
            }
        }

    def test_finalize_report_overall_fail_when_blocked(self):
        report = build_report_skeleton(Path("/tmp/wp0.yaml"), Path("/tmp/artifacts/wp0"), self.cfg, {"timestamp_utc": "x"})
        report["sections"] = {
            "a": {"status": STATUS_PASS},
            "b": {"status": STATUS_BLOCKED},
        }
        finalize_report(report)
        self.assertEqual(report["overall"]["result"], STATUS_FAIL)
        self.assertEqual(report["overall"]["counts"][STATUS_PASS], 1)
        self.assertEqual(report["overall"]["counts"][STATUS_BLOCKED], 1)

    def test_evaluate_approx_sync_pass(self):
        approx_tool = {
            "metrics": {
                "success_rate": 0.99,
                "pairs": 120,
                "slop_ms": 50.0,
                "queue_size": 10,
            }
        }
        out = evaluate_approx_sync(self.cfg, approx_tool)
        self.assertEqual(out["status"], STATUS_PASS)
        self.assertAlmostEqual(out["numeric_evidence"]["success_rate"], 0.99)

    def test_evaluate_approx_sync_blocked_without_tool(self):
        out = evaluate_approx_sync(self.cfg, None)
        self.assertEqual(out["status"], STATUS_BLOCKED)

    def test_evaluate_tray_stability_combines_pose_and_id(self):
        pose_tool = {"metrics": {"std_xyz_m": [0.001, 0.002, 0.0015], "gate": {"pass": True}}}
        id_tool = {"metrics": {"switch_rate": 0.005, "missing_rate": 0.02}}
        out = evaluate_tray_stability(self.cfg, pose_tool, id_tool)
        self.assertEqual(out["status"], STATUS_PASS)
        self.assertEqual(out["subchecks"]["missing_rate_reported"]["status"], STATUS_PASS)

    def test_evaluate_rosbag_replay_blocked_when_no_replay_metrics(self):
        rosbag_tool = {
            "metrics": {
                "record": {"command": ["ros2", "bag", "record", "--use-sim-time", "-o", "/tmp/bag", "/a"], "shell": "x"},
                "replay": {"command": ["ros2", "bag", "play", "/tmp/bag", "--clock"], "shell": "y"},
            }
        }
        out = evaluate_rosbag_replay(self.cfg, rosbag_tool, None, None)
        self.assertEqual(out["status"], STATUS_BLOCKED)
        self.assertEqual(out["subchecks"]["record_command_uses_sim_time"]["status"], STATUS_PASS)
        self.assertEqual(out["subchecks"]["replay_command_uses_clock"]["status"], STATUS_PASS)
        self.assertEqual(out["subchecks"]["replay_image_latency_p95_lt_120ms"]["status"], STATUS_BLOCKED)

    def test_evaluate_rosbag_replay_uses_only_remapped_topics(self):
        rosbag_tool = {
            "metrics": {
                "record": {"command": ["ros2", "bag", "record", "--use-sim-time", "-o", "/tmp/bag", "/a"], "shell": "x"},
                "replay": {"command": ["ros2", "bag", "play", "/tmp/bag", "--clock"], "shell": "y"},
            }
        }
        replay_image_tool = {
            "_replay": {
                "replay_topic_map": {
                    "/v5/cam/overhead/rgb": "/replay/v5/cam/overhead/rgb",
                    "/v5/cam/side/rgb": "/replay/v5/cam/side/rgb",
                }
            },
            "metrics": {
                "per_topic": {
                    "/replay/v5/cam/overhead/rgb": {"latency": {"p95_ms": 100.0}, "frames": 50},
                    "/replay/v5/cam/side/rgb": {"latency": {"p95_ms": 110.0}, "frames": 50},
                    "/v5/cam/overhead/rgb": {"latency": {"p95_ms": 999.0}, "frames": 1},
                }
            },
        }
        out = evaluate_rosbag_replay(self.cfg, rosbag_tool, replay_image_tool, "/tmp/bag")
        self.assertEqual(out["status"], STATUS_PASS)
        self.assertEqual(out["numeric_evidence"]["replay_image_latency_p95_ms"], 110.0)
        self.assertEqual(
            out["evidence"]["ignored_non_remapped_replay_topics"],
            ["/v5/cam/overhead/rgb"],
        )

    def test_evaluate_tray_stability_fail_on_id_switch_rate(self):
        pose_tool = {"metrics": {"std_xyz_m": [0.001, 0.001, 0.001], "gate": {"pass": True}}}
        id_tool = {"metrics": {"switch_rate": 0.02, "missing_rate": 0.0}}
        out = evaluate_tray_stability(self.cfg, pose_tool, id_tool)
        self.assertEqual(out["status"], STATUS_FAIL)
        self.assertEqual(out["subchecks"]["id_switch_lt_1pct"]["status"], STATUS_FAIL)

    def test_evaluate_tray_stability_blocked_on_id_switch_blocked_status(self):
        pose_tool = {"metrics": {"std_xyz_m": [0.001, 0.001, 0.001], "gate": {"pass": True}}}
        id_tool = {
            "status": STATUS_BLOCKED,
            "blocked_reason": "jsonl source unavailable",
            "metrics": {"source": {"mode": "jsonl", "path": "/tmp/v5_object_id_frames.jsonl"}},
        }
        out = evaluate_tray_stability(self.cfg, pose_tool, id_tool)
        self.assertEqual(out["status"], STATUS_BLOCKED)
        self.assertEqual(out["subchecks"]["id_switch_lt_1pct"]["status"], STATUS_BLOCKED)

    def test_attach_runner_metadata_preserves_runner_when_tool_json_missing(self):
        run = ToolRunResult(
            command=["python3", "-m", "hrl_trainer.v5.tools.tf_check_helper"],
            cwd="/tmp",
            returncode=1,
            stdout_tail="",
            stderr_tail="tf2 tools unavailable",
            json_output=None,
            error=None,
        )
        out = attach_runner_metadata(None, run)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["_runner"]["returncode"], 1)
        self.assertEqual(out["_runner"]["stderr_tail"], "tf2 tools unavailable")

    def test_evaluate_tf_contract_reports_runner_details_when_checks_missing(self):
        tf_tool = {
            "_runner": {
                "returncode": 1,
                "stderr_tail": "ros2: command not found",
            }
        }
        out = evaluate_tf_contract(self.cfg, tf_tool, Path("/tmp/artifacts/wp0/frames.pdf"))
        self.assertEqual(out["status"], STATUS_BLOCKED)
        self.assertIn("returncode=1", out["blocked_reason"])
        self.assertIn("stderr_tail=ros2: command not found", out["blocked_reason"])


if __name__ == "__main__":
    unittest.main()
