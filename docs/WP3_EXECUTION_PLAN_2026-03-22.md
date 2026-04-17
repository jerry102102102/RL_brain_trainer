# WP3 Execution Plan (2026-03-22)

## 0) Context and terminology (align with M2-9 closeout)
- **Current state (end of WP2):** We have a validated **real path** for benchmark/eval harness, but this is still not **real robot runtime / HIL** evidence.
- **real path (in this repo):** Pipeline can run end-to-end in repo/runtime environment (training/eval/benchmark orchestration and artifact generation), usually on simulation or offline dataset.
- **real robot runtime:** Controller/policy executes on physical or HIL-integrated runtime with hardware constraints (latency, actuator limits, watchdog, fail-safe), producing runtime logs with safety/rollback behavior.

---

## 1) WP3 Definition of Done (DoD)
WP3 is done only when all gates below are green:

1. **Runtime Gate (HIL/real robot):**
   - At least one repeatable HIL or real-robot execution path is scripted and reproducible.
   - Runtime logs include timing, safety events, and terminal status.
2. **Statistical Confidence Gate:**
   - Seed/episodes expanded from WP2 baseline to agreed target matrix.
   - Summary includes confidence interval / variance reporting and pass/fail thresholds.
3. **Deployment Safety Gate + Rollback Automation:**
   - Preflight safety checks block unsafe deployment.
   - Rollback can be triggered automatically on gate breach and leaves an auditable trail.
4. **Operational Readiness Gate:**
   - One-click (or one-command-chain) runbook exists for Jerry.
   - Evidence package (logs + metrics + gate report) can be generated and reviewed in < 30 minutes.

---

## 2) Workstreams and executable tasks

## WS1 — Real robot runtime / HIL gate
**Goal:** Close the gap between WP2 real path and production-like runtime behavior.

### WS1-T1: Define runtime contract and observability fields
- **Owner:** assistant
- **Input:** WP2 closeout notes, current benchmark/eval output schema
- **Output:** `docs/wp3/ws1_runtime_contract.md` (fields: timestamp, episode_id, action_latency_ms, safety_state, rollback_flag)
- **Acceptance command:**
  ```bash
  test -f docs/wp3/ws1_runtime_contract.md && echo "WS1-T1 PASS"
  ```

### WS1-T2: Prepare HIL run profile and environment checklist
- **Owner:** jerrineer
- **Input:** hardware topology, device endpoints, env vars, watchdog constraints
- **Output:** `configs/hil_runtime_profile.yaml` + `docs/wp3/hil_env_checklist.md`
- **Acceptance command:**
  ```bash
  test -f configs/hil_runtime_profile.yaml && test -f docs/wp3/hil_env_checklist.md && echo "WS1-T2 PASS"
  ```

### WS1-T3: Add runtime gate execution wrapper (non-core)
- **Owner:** assistant
- **Input:** runtime contract + HIL profile
- **Output:** `scripts/wp3/run_hil_gate.sh` (calls existing pipeline, no core algorithm changes)
- **Acceptance command:**
  ```bash
  bash -n scripts/wp3/run_hil_gate.sh && echo "WS1-T3 PASS"
  ```

### WS1-T4: First HIL dry-run + evidence capture
- **Owner:** Jerry
- **Input:** run wrapper + available test window
- **Output:** `artifacts/wp3/hil_dryrun/<date>/` logs + summary markdown
- **Acceptance command:**
  ```bash
  test -d artifacts/wp3/hil_dryrun && echo "WS1-T4 PASS"
  ```

---

## WS2 — Seed/episodes expansion and statistical confidence
**Goal:** Upgrade “single-run success” to statistically defensible result.

### WS2-T1: Lock experiment matrix (seeds × episodes × scenarios)
- **Owner:** assistant
- **Input:** WP2 baseline metrics, compute/time budget
- **Output:** `docs/wp3/ws2_experiment_matrix.md`
- **Acceptance command:**
  ```bash
  test -f docs/wp3/ws2_experiment_matrix.md && echo "WS2-T1 PASS"
  ```

### WS2-T2: Batch execution orchestration for expanded runs
- **Owner:** jerrineer
- **Input:** matrix spec, current eval harness entrypoints
- **Output:** `scripts/wp3/run_seed_episode_matrix.sh`
- **Acceptance command:**
  ```bash
  bash -n scripts/wp3/run_seed_episode_matrix.sh && echo "WS2-T2 PASS"
  ```

### WS2-T3: Confidence reporting template and threshold policy
- **Owner:** assistant
- **Input:** run outputs, acceptance policy
- **Output:** `docs/wp3/ws2_confidence_policy.md` + result template
- **Acceptance command:**
  ```bash
  test -f docs/wp3/ws2_confidence_policy.md && echo "WS2-T3 PASS"
  ```

### WS2-T4: Execute matrix and publish confidence summary
- **Owner:** Jerry
- **Input:** orchestrated runs complete
- **Output:** `artifacts/wp3/stat_report/<date>/summary.md` with CI/variance table
- **Acceptance command:**
  ```bash
  test -f artifacts/wp3/stat_report/README.md || echo "Create date-stamped summary under artifacts/wp3/stat_report/"
  ```

---

## WS3 — Deployment-grade safety gate + rollback automation
**Goal:** Make deployment fail-safe and auto-recoverable.

### WS3-T1: Safety gate checklist and block conditions
- **Owner:** assistant
- **Input:** runtime risk list, known failure modes
- **Output:** `docs/wp3/ws3_safety_gate.md` (hard-block vs soft-warning)
- **Acceptance command:**
  ```bash
  test -f docs/wp3/ws3_safety_gate.md && echo "WS3-T1 PASS"
  ```

### WS3-T2: Rollback trigger spec and implementation plan
- **Owner:** jerrineer
- **Input:** deployment mechanism, previous stable artifact reference
- **Output:** `docs/wp3/ws3_rollback_spec.md`
- **Acceptance command:**
  ```bash
  test -f docs/wp3/ws3_rollback_spec.md && echo "WS3-T2 PASS"
  ```

### WS3-T3: Safety + rollback runner script (non-core)
- **Owner:** assistant
- **Input:** gate spec + rollback spec
- **Output:** `scripts/wp3/run_safety_gate_and_rollback.sh`
- **Acceptance command:**
  ```bash
  bash -n scripts/wp3/run_safety_gate_and_rollback.sh && echo "WS3-T3 PASS"
  ```

### WS3-T4: Failure-injection rehearsal
- **Owner:** Jerry
- **Input:** controlled failure cases (latency spike, sensor timeout, policy divergence)
- **Output:** `artifacts/wp3/failure_rehearsal/<date>/` logs showing auto rollback fired
- **Acceptance command:**
  ```bash
  test -d artifacts/wp3/failure_rehearsal && echo "WS3-T4 PASS"
  ```

---

## WS4 — Integration, release gate, and operator runbook
**Goal:** Deliver one coherent WP3 gate and operator-ready execution path.

### WS4-T1: Consolidated WP3 gate command and output layout
- **Owner:** assistant
- **Input:** WS1~WS3 scripts/specs
- **Output:** `scripts/wp3/run_wp3_full_gate.sh` + `docs/wp3/output_layout.md`
- **Acceptance command:**
  ```bash
  bash -n scripts/wp3/run_wp3_full_gate.sh && test -f docs/wp3/output_layout.md && echo "WS4-T1 PASS"
  ```

### WS4-T2: Two-pass trial (shadow + candidate)
- **Owner:** Jerry
- **Input:** full gate script and candidate artifact
- **Output:** gate reports for pass A/B
- **Acceptance command:**
  ```bash
  test -d artifacts/wp3/gate_runs && echo "WS4-T2 PASS"
  ```

### WS4-T3: Go/No-Go meeting package
- **Owner:** jerrineer
- **Input:** all gate reports + risk/fallback status
- **Output:** `docs/wp3/WP3_GO_NO_GO_PACKET.md`
- **Acceptance command:**
  ```bash
  test -f docs/wp3/WP3_GO_NO_GO_PACKET.md && echo "WS4-T3 PASS"
  ```

---

## 3) Two-week sprint plan (Day 1 ~ Day 14)

- **Day 1:** Kickoff, align terminology, freeze DoD and gate thresholds.
- **Day 2:** WS1-T1 done; runtime contract reviewed.
- **Day 3:** WS1-T2 done; HIL env checklist closed.
- **Day 4:** WS1-T3 wrapper ready; dry parse and smoke check.
- **Day 5:** WS1-T4 first dry-run and evidence upload.
- **Day 6:** WS2-T1 matrix frozen; WS2-T2 scripting starts.
- **Day 7:** WS2-T2 complete; batch orchestration smoke pass.
- **Day 8:** WS2-T3 confidence policy complete.
- **Day 9:** WS2-T4 long-run execution (overnight) + preliminary stats.
- **Day 10:** WS3-T1 safety gate list frozen.
- **Day 11:** WS3-T2 rollback spec finalized.
- **Day 12:** WS3-T3 runner script + WS3-T4 rehearsal.
- **Day 13:** WS4-T1 integration command + WS4-T2 shadow run.
- **Day 14:** WS4-T3 Go/No-Go packet + decision meeting.

---

## 4) Risks and fallback strategy

1. **HIL slot or hardware instability risk**
   - **Impact:** WS1/WS3 timeline slip.
   - **Fallback:** Run simulated latency/failure injection with same gate scripts; keep HIL evidence as blocker for final Go.

2. **Experiment matrix runtime too long**
   - **Impact:** WS2 cannot finish in 2 weeks.
   - **Fallback:** Phase matrix into Tier-1 (must-have) and Tier-2 (optional), preserving CI-based decision on Tier-1.

3. **Rollback path not deterministic**
   - **Impact:** Safety gate not deployment-grade.
   - **Fallback:** enforce “no automatic deploy” mode, require manual rollback checklist until deterministic automation verified.

4. **Artifact schema inconsistency across WS**
   - **Impact:** hard to compile Go/No-Go packet.
   - **Fallback:** lock output layout in WS4-T1 and add preflight schema check command before every run.

---

## 5) Suggested gate command examples (directly runnable)

```bash
# WS1 quick check
bash -n scripts/wp3/run_hil_gate.sh

# WS2 quick check
bash -n scripts/wp3/run_seed_episode_matrix.sh

# WS3 quick check
bash -n scripts/wp3/run_safety_gate_and_rollback.sh

# WS4 final gate (to be created in WS4)
bash scripts/wp3/run_wp3_full_gate.sh
```

This document is planning-focused and intentionally avoids core algorithm/code modifications.
