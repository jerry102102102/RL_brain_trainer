import json
import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.benchmark_rule_l2_v0 import (
    BENCHMARK_SCHEMA,
    BENCHMARK_VERSION,
    load_rule_l2_v0_benchmark_summary,
    parse_rule_l2_v0_benchmark_summary,
    run_rule_l2_v0_benchmark,
    write_rule_l2_v0_benchmark_summary,
)


class TestV5BenchmarkRuleL2V0(unittest.TestCase):
    def test_parser_accepts_valid_payload(self):
        payload = {
            "schema": BENCHMARK_SCHEMA,
            "version": BENCHMARK_VERSION,
            "seed": 123,
            "episode_count": 4,
            "success_count": 3,
            "fail_count": 1,
            "average_reward": 1.25,
            "average_episode_length": 4.0,
        }
        summary = parse_rule_l2_v0_benchmark_summary(payload)
        self.assertEqual(summary.episode_count, 4)
        self.assertEqual(summary.success_count, 3)
        self.assertEqual(summary.fail_count, 1)

    def test_parser_rejects_missing_required_fields(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            parse_rule_l2_v0_benchmark_summary(
                {
                    "schema": BENCHMARK_SCHEMA,
                    "version": BENCHMARK_VERSION,
                }
            )

    def test_runner_is_deterministic_for_seed(self):
        summary_a = run_rule_l2_v0_benchmark(episodes=6, seed=99)
        summary_b = run_rule_l2_v0_benchmark(episodes=6, seed=99)
        self.assertEqual(summary_a.to_dict(), summary_b.to_dict())

    def test_write_and_load_summary_round_trip(self):
        summary = run_rule_l2_v0_benchmark(episodes=5, seed=17)
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "benchmark_summary.json"
            write_rule_l2_v0_benchmark_summary(summary, target)
            loaded = load_rule_l2_v0_benchmark_summary(target)

            self.assertEqual(loaded.to_dict(), summary.to_dict())
            # Ensure stable JSON key order formatting path remains parseable.
            text = target.read_text(encoding="utf-8")
            parsed = json.loads(text)
            self.assertEqual(parsed["schema"], BENCHMARK_SCHEMA)


if __name__ == "__main__":
    unittest.main()
