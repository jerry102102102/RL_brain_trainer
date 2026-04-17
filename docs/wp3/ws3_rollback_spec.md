# WS3 Rollback Spec (MVP)

Rollback target baseline policy: `rule_l2_v0`

## Command
```bash
bash scripts/wp3/run_safety_gate_and_rollback.sh --episodes 8 --seeds 42,43
```

## Why this changed
Previous WS3 gate used a single seed and tiny sample, which can produce brittle false negatives
(e.g. one unlucky run gets `success_count=0, fail_count=2` and blocks the whole WP3 pipeline).

WS3 now runs a **small multi-seed rollback check** and distinguishes:
- `pass`: all rollback seeds pass
- `conditional_pass`: policy/fallback invariants are enforced and at least one seed passes, but not all seeds pass
- `fail`: policy mismatch, fallback used, or every seed fails

This keeps the gate safety-conservative:
- We still fail hard on safety invariants (wrong policy or fallback).
- We avoid declaring hard fail from one unstable sample.
- We explicitly label degraded quality as `conditional_pass` instead of pretending everything is perfect.

## PASS criteria
Hard safety invariants (must be true):
- `policy_executed == rule_l2_v0` for every seed
- `fallback_used == false` for every seed

Outcome rule:
- `pass` when all seeds have `passed == true`
- `conditional_pass` when at least one seed passes but not all
- `fail` when no seeds pass (or safety invariants fail)

Evidence JSON is generated under `artifacts/reports/wp3/<timestamp>/ws3_rollback_gate.json`.
The JSON includes `gate_status`, `aggregate`, and per-seed details.
