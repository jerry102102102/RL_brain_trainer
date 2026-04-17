# WP2 Implementation Note (M2-7 ~ M2-9)

_Last updated: 2026-03-22_

This note aligns with:
- `/home/jerry/.openclaw/workspace/reports/WP2_M2_9_closeout_2026-03-22.md`
- `/home/jerry/.openclaw/workspace/reports/WP2_M2_9_one_page_summary_2026-03-22.md`

## Scope clarified (important)

- **Current stage is real path in benchmark/eval harness**:
  - `run_mode=real` means implemented v5 evaluation/benchmark code paths are executed (not placeholder/simulated rows).
- **This is NOT real robot runtime / HIL**:
  - No physical hardware execution proof is claimed in WP2 M2-9 package.

### Terminology
- **real path**: real execution path inside repository benchmark/eval harness (software path, deterministic test/eval flow).
- **real robot runtime**: physical robot / controller / HIL runtime on hardware.

## What WP2 completed (M2-7 ~ M2-9)

### M2-7
- Training loop integration evidence and tests in place.
- Backward compatibility + RL path counters + seed reproducibility verified by tests.

### M2-8 / M2-8a / M2-8b
- Baseline benchmark coverage passing.
- Eval harness runs both Rule/RL policy path with strict policy checks.
- Formal 4-variant comparison generated with `run_mode=real` for all variants.

### M2-9
- One-command rerun wrapper delivered:
  - `scripts/m2_9_rerun_wp2.sh`
- Closeout evidence package reproducible from single command.

## Completion checklist (status)

- [x] M2-7 integration tests green (`tests.test_v5_m2_7_training_loop_integration`)
- [x] M2-8 benchmark tests green (`tests.test_v5_benchmark_rule_l2_v0`)
- [x] M2-8a eval harness tests green (`tests.test_v5_eval_harness`)
- [x] M2-8b formal comparison tests green (`tests.test_v5_m2_8_formal_comparison`)
- [x] 4-variant formal comparison output generated (`run_mode=real`, `status=ok`)
- [x] One-click rerun script available and executable
- [ ] Physical robot / HIL validation (out of current WP2 package scope)

## Primary output files

- `artifacts/reports/v5/v5_eval_rule_l2_v0_seed42_ep8.json`
- `artifacts/reports/v5/v5_eval_rl_l2_seed11_ep4.json`
- `artifacts/reports/v5/v5_eval_rl_l2_seed13_ep4.json`
- `artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json`

## One-click rerun command

```bash
/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/scripts/m2_9_rerun_wp2.sh
```

## Minimal validation commands

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer

# test package
./scripts/m2_9_rerun_wp2.sh

# output existence
ls -1 artifacts/reports/v5/v5_eval_rule_l2_v0_seed42_ep8.json \
      artifacts/reports/v5/v5_eval_rl_l2_seed11_ep4.json \
      artifacts/reports/v5/v5_eval_rl_l2_seed13_ep4.json \
      artifacts/reports/v5/m2_8_formal_comparison_summary_seed42_ep8.json
```
