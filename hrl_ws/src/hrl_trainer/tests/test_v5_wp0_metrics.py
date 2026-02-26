import unittest

from hrl_trainer.v5.tools.metrics_core import (
    greedy_approx_sync_pairs_ns,
    summarize_id_switch,
    summarize_image_health,
    summarize_latency_ms,
    summarize_pose_jitter,
    summarize_state_topic_latency_by_topic,
)


class TestV5Wp0Metrics(unittest.TestCase):
    def test_latency_summary_with_gate(self):
        out = summarize_latency_ms([10, 20, 30, 40, 50], p95_limit_ms=60)
        self.assertEqual(out["count"], 5)
        self.assertTrue(out["gate"]["pass"])
        self.assertAlmostEqual(out["p50_ms"], 30.0)

    def test_image_health_drop_and_latency(self):
        # 10 Hz nominal over 5 emitted frames with one missing frame gap.
        header = [0, 100_000_000, 200_000_000, 400_000_000, 500_000_000]
        recv = [h + 20_000_000 for h in header]
        out = summarize_image_health(recv, header, expected_fps=10.0, latency_p95_limit_ms=120.0)
        self.assertEqual(out["frames"], 5)
        self.assertEqual(out["drop"]["drop_estimate_frames"], 1)
        self.assertTrue(out["latency"]["gate"]["pass"])

    def test_approx_sync_success_rate(self):
        left = [0, 100_000_000, 200_000_000, 300_000_000]
        right = [10_000_000, 105_000_000, 260_000_000]
        out = greedy_approx_sync_pairs_ns(left, right, slop_ms=50.0)
        self.assertEqual(out["pairs"], 3)
        self.assertAlmostEqual(out["success_rate"], 1.0)

    def test_pose_jitter_gate_and_aux(self):
        pts = [
            [1.0, 2.0, 3.0],
            [1.001, 2.0005, 2.9995],
            [0.9995, 1.9998, 3.0002],
        ]
        out = summarize_pose_jitter(pts, std_limit_m=0.003)
        self.assertTrue(out["gate"]["pass"])
        self.assertLess(out["radial_std_m"], 0.003)

    def test_id_switch_and_missing_rates(self):
        out = summarize_id_switch(["tray1", "tray1", None, "tray2", "tray2"], missing_warn_rate=0.05)
        self.assertEqual(out["switch_events"], 1)
        self.assertEqual(out["valid_frames"], 4)
        self.assertAlmostEqual(out["switch_rate"], 0.25)
        self.assertAlmostEqual(out["missing_rate"], 0.2)
        self.assertTrue(out["warnings"])

    def test_state_latency_summary_excludes_non_state_basis(self):
        out = summarize_state_topic_latency_by_topic({"/joint_states": [10, 20], "/foo": [30]}, p95_limit_ms=80.0)
        self.assertEqual(out["gate_basis"], "state_topics_only")
        self.assertTrue(out["overall"]["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
