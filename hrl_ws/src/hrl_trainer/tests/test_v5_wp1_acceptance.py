import unittest

from hrl_trainer.v5.pipeline import run_wp1_acceptance


class TestV5Wp1Acceptance(unittest.TestCase):
    def test_wp1_acceptance_smoke_and_random_summary(self):
        summary = run_wp1_acceptance(smoke_count=10, random_count=20, random_seed=42)

        self.assertIn("smoke", summary)
        self.assertIn("random", summary)
        self.assertIn("overall", summary)

        self.assertEqual(summary["smoke"]["task_count"], 10)
        self.assertEqual(summary["random"]["task_count"], 20)

        self.assertEqual(summary["overall"]["task_count"], 30)
        self.assertEqual(
            summary["overall"]["task_count"],
            summary["overall"]["success_count"] + summary["overall"]["fail_count"],
        )

        self.assertIsInstance(summary["overall"]["fail_reason_breakdown"], dict)


if __name__ == "__main__":
    unittest.main()
