import unittest

from hrl_trainer.v5.eval_harness import (
    EVAL_HARNESS_SCHEMA,
    EVAL_HARNESS_VERSION,
    POLICY_RL_L2,
    POLICY_RULE_L2_V0,
    parse_eval_harness_summary,
    require_policy_implementation,
    resolve_policy_execution,
    run_eval_harness,
)


class TestV5EvalHarness(unittest.TestCase):
    def test_parser_accepts_valid_payload(self):
        payload = {
            "schema": EVAL_HARNESS_SCHEMA,
            "version": EVAL_HARNESS_VERSION,
            "policy_requested": "rule_l2_v0",
            "policy_executed": "rule_l2_v0",
            "fallback_used": False,
            "seed": 42,
            "episodes": 8,
            "summary": {
                "stage": {
                    "name": "l2_policy_benchmark",
                    "benchmark_schema": "v5_rule_l2_v0_benchmark",
                    "benchmark_version": "1.0",
                },
                "reward": {"average_reward": 1.5, "average_episode_length": 4.0},
                "success_count": 6,
                "fail_count": 2,
            },
            "status_code": 0,
            "passed": True,
        }
        parsed = parse_eval_harness_summary(payload)
        self.assertEqual(parsed.schema, EVAL_HARNESS_SCHEMA)
        self.assertEqual(parsed.policy_executed, POLICY_RULE_L2_V0)

    def test_parser_rejects_missing_required_fields(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            parse_eval_harness_summary(
                {
                    "schema": EVAL_HARNESS_SCHEMA,
                    "version": EVAL_HARNESS_VERSION,
                }
            )

    def test_policy_selector_executes_rl_l2_without_fallback(self):
        executed, fallback = resolve_policy_execution(POLICY_RL_L2, strict_policy=False)
        self.assertEqual(executed, POLICY_RL_L2)
        self.assertFalse(fallback)

    def test_policy_selector_strict_mode_rejects_missing_implementation(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            resolve_policy_execution(POLICY_RL_L2, strict_policy=True, policy_runners={POLICY_RULE_L2_V0: object()})

    def test_guard_rejects_missing_policy_implementation(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            require_policy_implementation(POLICY_RL_L2, policy_runners={POLICY_RULE_L2_V0: object()})

    def test_strict_rl_l2_path_runs_and_emits_schema_output(self):
        summary = run_eval_harness(policy_requested=POLICY_RL_L2, episodes=4, seed=13, strict_policy=True)
        payload = summary.to_dict()
        self.assertEqual(payload["policy_requested"], POLICY_RL_L2)
        self.assertEqual(payload["policy_executed"], POLICY_RL_L2)
        self.assertFalse(payload["fallback_used"])
        self.assertEqual(payload["summary"]["stage"]["benchmark_schema"], "v5_rl_l2_v0_benchmark")
        parsed = parse_eval_harness_summary(payload)
        self.assertEqual(parsed.policy_executed, POLICY_RL_L2)

    def test_runner_is_deterministic_for_seed(self):
        summary_a = run_eval_harness(policy_requested="rule_l2_v0", episodes=6, seed=99)
        summary_b = run_eval_harness(policy_requested="rule_l2_v0", episodes=6, seed=99)
        self.assertEqual(summary_a.to_dict(), summary_b.to_dict())


if __name__ == "__main__":
    unittest.main()
