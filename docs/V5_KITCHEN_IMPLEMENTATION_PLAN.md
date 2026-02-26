# V5 Kitchen Manipulation — Three-Layer RL Implementation Plan

Status: Draft v1 (execution-ready)
Owner: Jerry + Assistant
Branch: `v5`
Repo: `repos/personal/RL_brain_trainer`

---

## 1) Goal (What we are building)

Build a deployable V5 pipeline that uses the RL brainer three-layer structure as the **primary control architecture** for kitchen manipulation.

User-level command is intentionally minimal:
- `MOVE_PLATE(source, target)`

System must autonomously handle:
- perception
- high-level subtask decomposition
- local skill decisions
- low-level arm control and safety

---

## 2) Success Criteria (Definition of Done)

A V5 build is considered done when all below are true:

1. **End-to-end execution works**
   - Given `MOVE_PLATE(A,B)`, system completes pick-and-place in Gazebo kitchen scene.

2. **Three-layer contract is enforced**
   - L1 does not output low-level control.
   - L2 outputs skill-level commands only.
   - L3 alone outputs executable arm trajectories.

3. **Metrics are reproducible**
   - Fixed config + seed can reproduce core metrics (success, collision, timeout, completion time).

4. **Safety is observable**
   - Safety intervention and failure reasons are logged per episode.

5. **Ablation results exist**
   - Rule-L2 baseline vs RL-L2 comparison on same tasks.

---

## 3) System Architecture (V5 mapping)

## 3.1 L1 — Perception + High-Level Task Reasoning
Input:
- scene state stream (`/tray_tracking/pose_stream` + robot state)
- user command (`MOVE_PLATE`)

Output (`IntentPacket`):
- `object_id`
- `pick_pose`
- `place_pose`
- `constraints` (clearance, speed caps, timeout)
- `subtask_graph`:
  - approach_pick
  - grasp
  - lift
  - transfer
  - place
  - retreat

Rules:
- no direct joint trajectory output
- only semantic/task-level outputs

## 3.2 L2 — Skill Policy / Local Planner (learning core)
Input:
- `IntentPacket`
- local arm/object state

Output (`SkillCommand`):
- `ee_target_pose` or trajectory chunk
- `gripper_cmd`
- current skill mode

Rules:
- can use memory/retrieval
- cannot bypass L3 safety layer

## 3.3 L3 — Deterministic Controller + Safety Shield
Input:
- `SkillCommand`
- controller feedback

Output:
- `/arm_controller/joint_trajectory`
- execution status + fail reason

Safety:
- joint limit check
- collision projection / stop-on-risk
- fallback (slowdown / halt / retreat)

---

## 4) Interface Contracts (concrete)

## 4.1 Topic / message contract (V5 minimum)
- `L1_out`: `/v5/intent_packet` (JSON/msg)
- `L2_out`: `/v5/skill_command`
- `L3_out`: `/arm_controller/joint_trajectory`
- `Camera streams (policy-visible)`:
  - `/v5/cam/overhead/rgb`
  - `/v5/cam/side/rgb`
  - (optional next) wrist cam `/v5/cam/wrist/rgb`
- `Robot state (policy-visible)`:
  - `/joint_states`
  - arm controller state topics
- `Object GT stream (reward/eval only)`:
  - `/tray_tracking/pose_stream`
  - `(to add) /tray1/pose` extracted single-object topic

## 4.2 Frame / sync contract
- Single canonical world frame for planning
- Fixed update rates:
  - L1: 1–2 Hz
  - L2: 10–20 Hz
  - L3: 50–200 Hz
- Timestamp validity check before command application

---

## 5) Implementation Work Packages

## WP0 — Environment and data path hardening
Deliverables:
- stable scene launch
- stable arm topic/control path
- multi-camera perception stream available
- stable tray single-object pose topic

Tasks:
- [ ] convert `/tray_tracking/pose_stream` into filtered `/tray1/pose`
- [ ] ensure deterministic naming and frame mapping
- [ ] add diagnostics script for pose stream health

Exit criteria:
- 5-minute continuous stream without dropouts
- one-command topic check script returns PASS

## WP1 — L1 task graph layer
Deliverables:
- command parser (`MOVE_PLATE`)
- deterministic subtask graph generator
- constraint packer

Tasks:
- [ ] implement `IntentPacket` schema
- [ ] implement slot→pose lookup
- [ ] implement failure path (`unreachable`, `missing object`)

Exit criteria:
- for 20 random tasks, valid intent packet generated with no low-level control fields

## WP2 — L2 baseline + RL slot
Deliverables:
- rule-based L2 baseline
- RL-L2 pluggable interface

Tasks:
- [ ] baseline policy for approach/grasp/place
- [ ] RL adapter respecting same `SkillCommand` schema
- [ ] memory hook (optional for phase-2)

Exit criteria:
- rule baseline can complete task in easy scene
- RL module can replace baseline without changing L1/L3

## WP3 — L3 execution and safety shield
Deliverables:
- deterministic trajectory executor
- safety shield and fallback

Tasks:
- [ ] command projection (joint limits, velocity, acceleration)
- [ ] collision-risk stop or slowdown
- [ ] structured fail reason logging

Exit criteria:
- no unsafe command reaches arm controller in stress tests

## WP4 — Evaluation harness
Deliverables:
- scenario suite (easy/medium/hard)
- metric logger + summary report

Tasks:
- [ ] benchmark scripts (`--seed`, `--episodes`)
- [ ] metrics JSON + markdown summary
- [ ] comparison tables: Rule-L2 vs RL-L2

Exit criteria:
- one-command benchmark run produces reproducible report

---

## 6) Experiment Matrix

E1. Integration baseline (Rule L2)
- Purpose: verify end-to-end contract correctness
- Metrics: success, collision, timeout

E2. RL-L2 replacement
- Purpose: show learning benefit over baseline
- Metrics: success/time tradeoff

E3. Complexity scaling
- easy: clear scene
- medium: mild clutter / varied start pose
- hard: constrained workspace + distractor objects

E4. Robustness checks
- perception noise injection
- control latency injection
- domain randomization

---

## 7) Metrics and Reporting Format

Core metrics:
- `success_rate`
- `collision_rate`
- `timeout_rate`
- `mean_completion_time`
- `safety_intervention_rate`
- `replan_count_l1/l2`

Per-run artifacts:
- config hash
- git commit hash
- seed list
- result json path
- failure case snapshots

Report template (each run):
1. Setup
2. Metrics
3. Failure breakdown
4. One conclusion
5. Next single change

---

## 8) Timeline (practical)

Sprint A (1–2 days): WP0 + WP1 skeleton
- stable topic/data + intent packet

Sprint B (2–3 days): WP2 + WP3 baseline
- complete rule-L2 end-to-end execution with safety

Sprint C (2–3 days): WP4 + RL-L2 swap
- run benchmark and produce first comparison report

---

## 9) Risks and Mitigations

Risk: perception stream ambiguity (multi-object Pose_V)
- Mitigation: explicit tray selector and single-topic extraction

Risk: policy commands violate constraints
- Mitigation: strict L3 projection and hard fail-safe

Risk: hard to debug failures
- Mitigation: typed fail reasons + replayable run logs

Risk: overfitting to one layout
- Mitigation: scene randomization + holdout test set

---

## 10) Immediate Next Actions (today)

1. Finalize `/tray1/pose` extraction node from current pose stream.
2. Implement `IntentPacket` schema and `MOVE_PLATE(A,B)` parser.
3. Stub `SkillCommand` pipeline and connect L3 executor path.
4. Run first end-to-end dry run with Rule-L2 in easy scene.

---

## 11) File/Code Placement Plan

Suggested additions:
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/intent_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/skill_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/control_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/pipeline.py`
- `hrl_ws/src/hrl_trainer/config/v5_kitchen_*.yaml`
- `docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` (this file)
- `docs/V5_EXPERIMENT_LOG.md`

---

## 11.1 Observation boundary policy (must-follow)

Policy-visible (can be fed into L1/L2 model):
- camera RGB streams (`/v5/cam/overhead/rgb`, `/v5/cam/side/rgb`)
- robot state (`/joint_states`, arm controller state)
- task command and stage flags

Policy-hidden (must NOT be fed to policy):
- tray ground-truth pose stream (`/tray_tracking/pose_stream`)
- derived tray GT labels used for reward and success checks

Use of tray GT is restricted to:
- reward calculation
- terminal success/failure judgment
- offline evaluation/analysis

## 12) Acceptance Gate for planning handoff

Before writing the final formal proposal, confirm:
- [ ] task contract accepted (`MOVE_PLATE` only)
- [ ] three-layer boundaries accepted
- [ ] metrics set accepted
- [ ] timeline accepted
- [ ] first sprint scope accepted

Once these are confirmed, this plan can be directly transformed into the final project plan/proposal.
