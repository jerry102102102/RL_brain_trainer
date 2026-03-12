import unittest
from unittest import mock

from hrl_trainer.v5.tools import runtime_parity_check
from hrl_trainer.v5.tools.runtime_parity_check import (
    REQUIRED_TOPICS,
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_PASS,
    TopicProbeResult,
    _build_parity_report,
    _parse_launch_cmd,
    _path_status,
)


class TestRuntimeParityCheck(unittest.TestCase):
    def test_parse_launch_cmd_uses_default_script(self):
        cmd = _parse_launch_cmd(None)
        self.assertTrue(cmd[-1].endswith("/scripts/v5/launch_kitchen_scene.sh"))

    def test_path_status_pass_when_all_topics_have_samples(self):
        rows = [TopicProbeResult(topic=t, listed=True, sample=True, elapsed_sec=0.1) for t in REQUIRED_TOPICS]
        self.assertEqual(_path_status(rows), STATUS_PASS)

    def test_path_status_fail_when_one_topic_missing_sample(self):
        rows = [TopicProbeResult(topic=t, listed=True, sample=True, elapsed_sec=0.1) for t in REQUIRED_TOPICS]
        rows[0] = TopicProbeResult(topic=REQUIRED_TOPICS[0], listed=True, sample=False, elapsed_sec=0.1)
        self.assertEqual(_path_status(rows), STATUS_FAIL)

    def test_build_parity_report_pass_for_matching_samples(self):
        manual = {
            "status": STATUS_PASS,
            "topics": [{"topic": t, "listed": True, "sample": True, "elapsed_sec": 0.1} for t in REQUIRED_TOPICS],
        }
        auto = {
            "status": STATUS_PASS,
            "topics": [{"topic": t, "listed": True, "sample": True, "elapsed_sec": 0.2} for t in REQUIRED_TOPICS],
        }
        parity = _build_parity_report(manual, auto)
        self.assertEqual(parity["status"], STATUS_PASS)
        self.assertEqual(parity["reason"], "all_topics_match")

    def test_build_parity_report_fail_for_sample_mismatch(self):
        manual = {
            "status": STATUS_PASS,
            "topics": [{"topic": t, "listed": True, "sample": True, "elapsed_sec": 0.1} for t in REQUIRED_TOPICS],
        }
        auto_topics = [{"topic": t, "listed": True, "sample": True, "elapsed_sec": 0.2} for t in REQUIRED_TOPICS]
        auto_topics[-1]["sample"] = False
        auto = {"status": STATUS_FAIL, "topics": auto_topics}
        parity = _build_parity_report(manual, auto)
        self.assertEqual(parity["status"], STATUS_FAIL)
        self.assertEqual(parity["per_topic"][REQUIRED_TOPICS[-1]]["match"], False)

    def test_build_parity_report_blocked_when_manual_blocked(self):
        manual = {"status": STATUS_BLOCKED, "blocked_reason": "not_requested", "topics": []}
        auto = {"status": STATUS_PASS, "topics": []}
        parity = _build_parity_report(manual, auto)
        self.assertEqual(parity["status"], STATUS_BLOCKED)
        self.assertIn("manual_blocked", parity["reason"])

    def test_main_auto_mode_returns_fail_when_launch_binary_missing(self):
        with mock.patch.object(runtime_parity_check, "_parse_args") as parse_args, mock.patch.object(
            runtime_parity_check, "_parse_launch_cmd", return_value=["/missing/launch.sh"]
        ), mock.patch.object(runtime_parity_check, "write_json"):
            parse_args.return_value = mock.Mock(
                mode="auto",
                timeout_sec=0.01,
                auto_launch_cmd=None,
                no_kill_before_auto=True,
                output=None,
                no_pretty=True,
            )
            rc = runtime_parity_check.main()
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
