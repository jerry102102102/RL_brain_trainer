# V5 Kitchen Manipulation — Three-Layer RL Execution Plan

Status: Draft v2 (critique-integrated)
Owner: Jerry + Assistant
Branch: `v5`
Repo: `repos/personal/RL_brain_trainer`

---

## 1) Goal

Build a deployable V5 pipeline where the three-layer RL brainer is the **primary architecture** for kitchen manipulation.

User command stays minimal:
- `MOVE_PLATE(source_slot, target_slot)`

System handles all downstream work:
- perception
- high-level decomposition
- local skill decision
- low-level arm execution + safety

---

## 2) Critical Design Decision (fixing ambiguity)

We explicitly split into two phases:

### Phase-0: GT-chain validation (control chain validation)
Purpose:
- validate L1/L2/L3 contract and execution stability quickly.

Policy-visible:
- robot state
- task/stage flags
- **GT-derived object estimate proxy allowed only in this phase**

Policy-hidden:
- raw tray GT stream remains hidden as direct training target.

### Phase-1: Vision-first validation (target V5)
Purpose:
- validate three-layer framework in realistic observation setting.

Policy-visible:
- camera streams + robot state + stage flags
- optional perception output topic `/v5/perception/object_pose_est`

Policy-hidden:
- `/tray_tracking/pose_stream` and tray GT derivatives (reward/eval only)

> V5 completion is judged on **Phase-1**, not only Phase-0.

---

## 3) Success Criteria (DoD)

1. End-to-end `MOVE_PLATE(A,B)` works in kitchen scene.
2. Three-layer boundaries remain clean:
   - L1 no low-level control output.
   - L2 no trajectory points/spline chunks.
   - L3 exclusively owns executable trajectory output.
3. Reproducible metrics under fixed config + seed.
4. Safety interventions and fail reasons are structured and queryable.
5. Rule-L2 vs RL-L2 ablation report exists in same benchmark suite.
6. Phase-1 vision-first experiment passes minimum acceptance gates.

---

## 4) System Architecture (V5 mapping)

## 4.1 L1 — Perception + High-Level Reasoning
Input (Phase-1 target):
- camera streams (`/v5/cam/overhead/rgb`, `/v5/cam/side/rgb`, optional wrist)
- robot state
- task command + stage flags
- perception module output `/v5/perception/object_pose_est` (policy-visible pose estimate)

Output (`IntentPacket`):
- `object_id`
- `pick_pose_candidates: [pose...]`
- `place_pose_candidates: [pose...]`
- `constraints` (clearance, speed caps, timeout)
- `reachability_hint`: `{ik_feasible, min_clearance_est, preferred_approach}`
- `grasp_hint`: `{pregrasp_offset, approach_axis, wrist_yaw_range}`
- `subtask_graph` (with recovery edges; see §4.4)

Rules:
- no direct control output
- no direct use of GT pose stream in Phase-1

## 4.2 L2 — Skill Policy / Local Planner
Input:
- `IntentPacket`
- robot state + perception estimate

Output (`SkillCommand`) — **strict schema**:
- `skill_mode` ∈ {APPROACH, GRASP, LIFT, TRANSFER, PLACE, RETREAT}
- `ee_target_pose` **or** `delta_pose`
- `gripper_cmd` ∈ {OPEN, CLOSE, HOLD}
- `speed_profile_id` ∈ {SLOW, NORMAL}
- `guard`:
  - `keep_level`
  - `max_tilt`
  - `min_clearance`

Hard prohibition:
- no time-parameterized trajectory chunk
- no spline points
- no direct `JointTrajectory` output
- no trajectory chunk / spline / time-parameterized command fields

## 4.3 L3 — Deterministic Controller + Safety Shield
Input:
- `SkillCommand`
- controller feedback

Output:
- `/arm_controller/joint_trajectory`
- `execution_status`
- `intervention_log`

---

## 5) Interface Contracts

## 5.1 Topic contract (minimum)
- `L1_out`: `/v5/intent_packet`
- `L2_out`: `/v5/skill_command`
- `L3_out`: `/arm_controller/joint_trajectory`
- `Perception_out`: `/v5/perception/object_pose_est` (policy-visible estimate)

Policy-visible streams:
- `/v5/cam/overhead/rgb`
- `/v5/cam/side/rgb`
- `/joint_states`
- arm controller state topics

Policy-hidden streams:
- `/tray_tracking/pose_stream`
- `/tray1/pose` (if derived from GT)

GT-only usage:
- reward
- terminal success/fail judgment
- offline eval and diagnostics

## 5.2 Camera Contract (hard spec)
- Message type: `sensor_msgs/msg/Image`
- Encoding: `rgb8` (fixed)
- Resolution: `640x480` (fixed in v1)
- FPS target:
  - overhead: 10 Hz
  - side: 10 Hz
  - optional wrist: 15 Hz
- Time base: sim time
- Sync strategy:
  - Phase-0: single-cam or async allowed
  - Phase-1: approximate sync (`overhead + side`)
- Camera info required:
  - `/v5/cam/overhead/camera_info`
  - `/v5/cam/side/camera_info`
- Frame contract:
  - `world -> cam_*_link -> cam_*_optical_frame` must exist in TF
- Latency budget:
  - image pipeline P95 < 120 ms

## 5.3 Frame / rate contract
- Canonical world frame for planning
- update rates:
  - L1: 1–2 Hz
  - L2: 10–20 Hz
  - L3: 50–200 Hz
- stale timestamp guard before execution

## 5.4 SafetySpec (measurable)
Inputs:
- robot state (joint/vel/acc)
- planning scene / collision geometry
- object pose estimate + uncertainty
- optional depth/occupancy cues

Checks and triggers:
- hard bounds violated (joint/vel/acc) -> `HALT`
- collision distance `< d_stop` -> `HALT`
- collision distance `< d_slow` -> `SLOWDOWN`
- tilt `> max_tilt` -> `HALT + RETREAT`

Outputs (structured):
- `safety_intervention: {type, timestamp, reason, suggested_recover_action}`
- `fail_reason` (typed enum in §8)

Intervention counting rule:
- count one intervention whenever `type` enters non-`NONE` state from previous control cycle.

---

## 6) Observation Pipeline (policy-visible)

Image preprocessing contract:
- resize: `224x224` (default)
- normalization: `[0,1]` then channel-wise norm
- frame stack: `N=2` (default)

Encoder contract:
- Phase-0: lightweight CNN encoder allowed
- Phase-1: frozen visual encoder preferred (CNN/ViT) with latent output

Policy observation vector:
- `obs_latent` (e.g., 256-d)
- `robot_state`
- `stage_flag`
- optional `object_pose_est` (from perception module, not GT)

GT never enters policy tensor in Phase-1.

---

## 7) SlotMap Specification (for MOVE_PLATE)

`SlotMap` schema:
- `slot_id: str`
- `region_world: {center_xyz, size_xyz, yaw}`
- `approach_pose_candidates: [pose...]`
- `place_pose_candidates: [pose...]`
- `allowed_objects: [id...]`
- `priority: int`

Task resolution rules:
- `MOVE_PLATE(A,B)` resolves through SlotMap only
- ambiguity triggers `TASK_DISAMBIGUATION_REQUIRED`

## 7.1 Recovery / Retry policy
Each subtask must define fail transitions:
- `APPROACH` fail -> re-approach (adjust yaw/offset) or `RETREAT`
- `GRASP` fail -> retry with updated grasp hint (within budget)
- `TRANSFER` risk stop -> `RETREAT` then replan
- `PLACE` fail -> micro-adjust then retry, else fallback pose

Required fields:
- `recovery_edges` in subtask graph
- `retry_budget` per skill stage
- `fallback_goal_pose` (safe hover/home)

---

## 8) Safety Semantics (structured)

`InterventionType` enum:
- `SLOWDOWN`
- `HALT`
- `RETREAT`
- `PROJECTION_CLAMP`

`FailReason` enum:
- `IK_FAIL`
- `COLLISION_PREDICT`
- `CONTACT`
- `OBJECT_LOST`
- `GRASP_FAIL`
- `PLACE_FAIL`
- `TIMEOUT`
- `SYNC_STALE`

Each intervention must emit structured log:
- timestamp
- layer source
- type
- trigger metric
- recovery action

---

## 9) Work Packages

## WP0 — Scene/Perception infrastructure hardening
Deliverables:
- stable scene launch
- stable arm control path
- stable multi-camera streams
- rosbag record/replay pipeline

Tasks:
- [ ] camera SDF/launch integration verified in scene
- [ ] `/v5/cam/*/rgb` + `/camera_info` publishing
- [ ] TF check script (`view_frames` / `tf2_echo`) passes
- [ ] image health diagnostics (fps, dropped frame, latency)
- [ ] rosbag2 record/replay script for camera+state topics
- [ ] tray stream extraction (`/tray_tracking/pose_stream` -> `/tray1/pose` GT-only)

Exit criteria:
- 5-min stable camera + state capture
- timestamp latency P95 < 80 ms
- static-scene pose jitter std < 3 mm
- object-id switch rate < 1% (or deterministic selector documented)
- replay reproduces same topic structure

## WP1 — L1 layer (phase-aware)
Deliverables:
- `MOVE_PLATE` parser
- SlotMap resolver
- IntentPacket generator
- perception adapter output `/v5/perception/object_pose_est`

Tasks:
- [ ] IntentPacket schema
- [ ] SlotMap implementation
- [ ] Phase-0 GT proxy switch + Phase-1 vision-only switch
- [ ] failure path (`unreachable`, `missing object`, `disambiguation required`)

Exit criteria:
- 20 random tasks produce valid intent packets with clean layer boundary

## WP2 — L2 baseline + RL slot
Deliverables:
- Rule-L2 baseline
- RL-L2 pluggable module

Tasks:
- [ ] SkillCommand strict schema enforcement
- [ ] baseline skill policy (approach/grasp/place)
- [ ] RL adapter consuming observation pipeline latent
- [ ] memory hook (optional)

Exit criteria:
- RL module can replace rule baseline without changing L1/L3 APIs

## WP3 — L3 execution + safety
Deliverables:
- deterministic executor
- safety shield
- structured intervention/failure logs

Tasks:
- [ ] projection to safe command space
- [ ] intervention triggers + recovery policy
- [ ] enum-based structured logs

Exit criteria:
- no unsafe command reaches controller in stress scenarios

## WP4 — Evaluation harness
Deliverables:
- benchmark suite (easy/medium/hard)
- report generator

Tasks:
- [ ] reproducible benchmark CLI (`--seed --episodes`)
- [ ] Rule-L2 vs RL-L2 report
- [ ] Phase-0 vs Phase-1 comparison report

Exit criteria:
- one-command run outputs metrics + failure breakdown + artifacts

---

## 10) Experiment Matrix

E0 (Phase-0): GT-chain validation
- objective: validate control chain and contracts

E1 (Phase-1): Vision-only baseline
- objective: validate perception-driven three-layer behavior

E2: RL-L2 vs Rule-L2
- objective: quantify policy value

E3: Complexity scaling
- easy/medium/hard scene sets

E4: Robustness
- image noise, latency injection, domain randomization

E5: Holdout generalization (hard requirement)
- Holdout layouts: 2–3 unseen kitchen layouts
- Holdout objects: variation in plate size/mass/friction/CoM
- Holdout camera conditions: lighting/camera pose variants
- Deliverables: `train_set.yaml`, `eval_set.yaml`, fixed seed lists

---

## 10.1 RL-L2 training loop contract
Observation (policy-visible):
- image latent(s) from observation pipeline
- robot state
- stage flag
- optional perception estimate (`/v5/perception/object_pose_est`)

Action space (learned fields in SkillCommand):
- `skill_mode` switch policy (optional staged)
- `delta_pose` / `ee_target_pose` selection
- `speed_profile_id`
- guard parameter selection within bounded range

Reward components:
- success bonus
- step/time penalty
- collision proximity penalty
- intervention penalty
- placement quality bonus/penalty

Termination:
- success
- hard safety failure
- timeout
- object lost (configurable recoverable terminal)

Curriculum:
- easy -> medium -> hard progression tied to success threshold

## 11) Metrics

Core:
- `success_rate`
- `collision_rate`
- `timeout_rate`
- `mean_completion_time`
- `safety_intervention_rate`
- `replan_count_l1`
- `replan_count_l2`

Per-run artifact:
- config hash
- commit hash
- seeds
- metrics json
- replay bag path
- failure snapshots

---

## 12) Timeline (realistic)

### Phase-0 (GT-chain) — ~1 week
- WP0 + WP1 minimal + WP3 baseline + E0

### Phase-1 (Vision-only) — +1 to 2 weeks
- camera contract hardening
- observation pipeline/encoder
- RL-L2 training + E1/E2/E3/E4

> Previous 1–2 day sprint estimate is retained only for partial infra tasks, not full vision-first validation.

---

## 13) Immediate Next Actions

1. Add `/camera_info` and TF verification scripts.
2. Add `/v5/perception/object_pose_est` adapter stub.
3. Enforce SkillCommand strict schema (remove trajectory chunk path).
4. Implement SlotMap spec and 10-task smoke test.
5. Add rosbag2 record/replay command set.

---

## 14) File/Code Placement

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/intent_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/perception_adapter.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/skill_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/control_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/pipeline.py`
- `hrl_ws/src/hrl_trainer/config/v5_kitchen_*.yaml`
- `docs/V5_EXPERIMENT_LOG.md`

---

## 15) Acceptance Gate

- [ ] Phase split accepted (Phase-0 GT / Phase-1 Vision)
- [ ] L2 strict output boundary accepted
- [ ] camera contract accepted
- [ ] observation pipeline accepted
- [ ] SlotMap spec accepted
- [ ] safety enums/logging accepted
- [ ] timeline accepted

Once these are accepted, convert directly into final proposal/execution handbook.
