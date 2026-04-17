import tempfile
import unittest
from pathlib import Path

from hrl_trainer.v5.benchmark_m2_8_formal_comparison import (
    BENCHMARK_SCHEMA,
    BENCHMARK_VERSION,
    PRIMARY_VARIANT,
    RUN_MODE_REAL,
    RUN_MODE_SIMULATED,
    VARIANT_LABELS,
    load_m2_8_formal_comparison_summary,
    parse_m2_8_formal_comparison_summary,
    run_m2_8_formal_comparison,
    write_m2_8_formal_comparison_summary,
)


class TestV5M28FormalComparison(unittest.TestCase):
    def test_summary_schema_and_variant_row_completeness(self):
        summary = run_m2_8_formal_comparison(episodes=5, seed=123)
        payload = summary.to_dict()

        self.assertEqual(payload["schema"], BENCHMARK_SCHEMA)
        self.assertEqual(payload["version"], BENCHMARK_VERSION)
        self.assertEqual(payload["primary_variant"], PRIMARY_VARIANT)
        self.assertTrue(payload["deterministic_seeded_runs"])

        rows = payload["variants"]
        self.assertEqual([row["label"] for row in rows], list(VARIANT_LABELS))
        required_row_fields = {
            "label",
            "status",
            "success_count",
            "fail_count",
            "avg_reward",
            "avg_episode_len",
            "collision_proxy",
            "run_mode",
        }
        for row in rows:
            self.assertEqual(set(row.keys()), required_row_fields)

        self.assertFalse([row for row in rows if row["run_mode"] == RUN_MODE_SIMULATED])
        for row in rows:
            self.assertEqual(row["run_mode"], RUN_MODE_REAL)
            self.assertEqual(row["status"], "ok")
            self.assertNotIn("placeholder", row["status"])

        metadata = payload["metadata"]
        self.assertFalse(metadata["contains_simulated_variants"])
        self.assertEqual(metadata["simulated_variant_labels"], [])
        self.assertEqual(sorted(metadata["real_variant_labels"]), sorted(list(VARIANT_LABELS)))
        for label in VARIANT_LABELS:
            self.assertEqual(metadata["variant_execution_modes"][label], RUN_MODE_REAL)

    def test_parser_rejects_incomplete_variant_row(self):
        payload = run_m2_8_formal_comparison(episodes=4, seed=7).to_dict()
        del payload["variants"][0]["run_mode"]

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            parse_m2_8_formal_comparison_summary(payload)

    def test_runner_is_deterministic_for_same_seed(self):
        summary_a = run_m2_8_formal_comparison(episodes=6, seed=99)
        summary_b = run_m2_8_formal_comparison(episodes=6, seed=99)
        self.assertEqual(summary_a.to_dict(), summary_b.to_dict())

    def test_write_and_load_round_trip(self):
        summary = run_m2_8_formal_comparison(episodes=6, seed=11)
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "m2_8_summary.json"
            write_m2_8_formal_comparison_summary(summary, target)
            loaded = load_m2_8_formal_comparison_summary(target)
            self.assertEqual(loaded.to_dict(), summary.to_dict())


if __name__ == "__main__":
    unittest.main()
