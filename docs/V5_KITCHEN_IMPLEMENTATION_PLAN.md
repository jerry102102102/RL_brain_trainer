# V5 Kitchen Manipulation — Three-Layer RL Execution Plan

Reference doctrine: `docs/V5_DESIGN_PHILOSOPHY.md` (L1/L2/L3 separation, L2-first shaping, simulation-first).

Status: Draft v2 (critique-integrated) + WP0 PASS (6/6, with replay) as of 2026-03-09
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
- canonical GT upstream `/tray_tracking/pose_stream_raw` (policy-hidden; reward/eval only)
- legacy `/tray_tracking/pose_stream` disabled by default
- tray GT derivatives (reward/eval only)

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
- `PoseCandidate = {pose, score, approach_axis, pregrasp_offset, pos_std, yaw_std}`
- `object_id`
- `pick_pose_candidates: [PoseCandidate...]`
- `place_pose_candidates: [PoseCandidate...]`
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
- `skill_mode` ∈ {APPROACH, INSERT_SUPPORT, LIFT_CARRY, PLACE, WITHDRAW, RETREAT}
- `ee_target_pose` **or** `delta_pose`
- `u_slot_params`:
  - `insert_depth`
  - `lateral_alignment`
  - `vertical_clearance`
  - `entry_yaw`
- `timing_params`:
  - `approach_speed_scale`
  - `lift_profile_id`
  - `contact_settle_time`
- `guard`:
  - `keep_level`
  - `max_tilt`
  - `min_clearance`
  - `fragility_mode_hint`

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
- `Perception_out`: `/v5/perception/object_pose_est` (policy-visible estimate, `ObjectPoseArray`)

### 5.1.a Canonical pipeline definitions（來源 -> 去向）

#### Pipeline A: Runtime perception（online 主線）
1. Gazebo world pose source（raw）
   - `/world/empty/dynamic_pose/info` (`gz.msgs.Pose_V`)
2. ROS bridge remap（name-preserving）
   - `/tray_tracking/pose_stream_raw` (`ros_gz_interfaces/msg/Pose_V`)
3. Tray extractor（deterministic）
   - `/tray1/pose` (`geometry_msgs/msg/PoseStamped`, frame=`world`)
4. Object id publisher / perception adapter
   - `/v5/perception/object_pose_est`（給 L1/L2 policy-visible）

**Direction:** Gazebo -> Bridge -> Extractor -> Perception_out -> L1/L2

#### Pipeline B: Three-layer execution（control 主線）
1. L1 intent generation
   - output: `/v5/intent_packet`
2. L2 policy / local planner
   - input: intent + policy-visible obs
   - output: `/v5/skill_command`
3. L3 deterministic executor + safety shield
   - input: skill command + controller feedback
   - output: `/arm_controller/joint_trajectory`

**Direction:** L1_out -> L2_out -> L3_out -> robot controller

#### Pipeline C: Training/diagnostics（offline/analysis 支線）
1. Obs builder（camera + robot state + perception）
2. Reward composer（sparse + PBRS + safety + smooth）
3. Rollout logger / replay integrity checker
4. Benchmark evaluator（Rule-L2 vs RL-L2）

**Direction:** Runtime logs/topics -> training artifacts -> evaluation reports

#### Legacy path status（明確標示）
- `/tray_tracking/pose_stream` 僅為 legacy/disabled-by-default 參考路徑，不作為 WP2 主線輸入。

#### Jazzy Pose_V limitation + degraded-mode policy
- 已確認在當前 Jazzy 環境：
  - `ros_gz_interfaces.msg` 無可用 `Pose_V` Python 型別（dedicated extractor import 會失敗）。
  - `ros_gz_bridge` 對 `/world/empty/dynamic_pose/info @ ros_gz_interfaces/msg/Pose_V` 缺少 template specialization。
- Runtime/launch 切換規則：
  - `enable_dedicated_tray_source:=true` 且 capability 支援 -> `tray_pose_mode=dedicated`
  - `enable_dedicated_tray_source:=true` 且 capability 不支援，且 `auto_fallback_legacy_on_unsupported:=true` -> `tray_pose_mode=legacy_degraded`（自動啟用 legacy adapter）
  - `enable_dedicated_tray_source:=false` + `enable_legacy_tray_pose_adapter:=true` -> 強制 legacy
- 強制 dedicated（僅在環境支援時）:
  - `enable_dedicated_tray_source:=true auto_fallback_legacy_on_unsupported:=false`

### 5.1.b Real-world deployment pipelines（真實環境）

> 原則：真實環境沿用同一份 L1/L2/L3 契約，不改語義介面，只替換資料來源與控制落地端。

#### Pipeline D: Real perception（online）
1. 實體感測器輸入
   - RGB / depth / robot proprioception /（可選）force-torque
2. 感知與追蹤
   - object detection + pose estimation + confidence/staleness gate
3. Policy-visible perception output（與 sim 對齊）
   - `/v5/perception/object_pose_est`
4. 餵給 L1/L2
   - L1 intent construction
   - L2 skill policy observation

**Direction:** sensors -> perception stack -> `/v5/perception/object_pose_est` -> L1/L2

#### Pipeline E: Real execution（control）
1. L1
   - output: `/v5/intent_packet`
2. L2
   - output: `/v5/skill_command`
3. L3（real controller adapter）
   - 將 skill command 投影到硬體可執行控制命令（joint/cartesian/servo）
4. Hardware control loop
   - robot driver / controller manager / actuator
5. Feedback + safety telemetry
   - execution status / intervention log / fail reason

**Direction:** L1_out -> L2_out -> L3(real adapter) -> hardware controller -> robot

#### Pipeline F: Real safety & supervision（必備）
1. Online safety checks
   - collision distance / tilt / joint bounds / workspace forbidden zones / e-stop state
2. Intervention manager
   - HALT / SLOWDOWN / RETREAT / HOLD
3. Structured logs
   - intervention type, trigger metric, timestamp, recover action
4. Human override path
   - manual takeover / kill-switch / resume policy

**Direction:** controller feedback + environment context -> safety shield -> intervention + logs

#### Pipeline G: Sim-to-Real evaluation（一致性管線）
1. Contract parity check
   - L1/L2/L3 topic schema parity（sim vs real）
2. Metric parity check
   - success/collision/time/intervention 指標定義一致
3. Replayability artifacts
   - real-run logs 可回放成同格式評估報告
4. Deployment gate
   - 未通過 parity gate 不可進 production rollout

**Direction:** sim metrics + real metrics -> parity report -> deploy/hold decision

ObjectPoseArray contract:
- `header.frame_id = world` (fixed)
- `objects[] = {object_id, pose, confidence, pos_std, yaw_std, stamp}`
- confidence/staleness gate: if `confidence < tau_conf` or `now - stamp > dt_stale`, L1 must emit `missing object` path

Policy-visible streams:
- `/v5/cam/overhead/rgb`
- `/v5/cam/side/rgb`
- `/joint_states`
- arm controller state topics

Policy-hidden streams:
- `/tray_tracking/pose_stream_raw` (canonical GT bridge stream)
- `/tray_tracking/pose_stream` (legacy, disabled by default)
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
- TTC `< ttc_halt` -> `HALT`

Default trigger thresholds (v1):
- `d_slow = 0.02 m`
- `d_stop = 0.01 m`
- `max_tilt = 5 deg`
- `ttc_halt = 0.3 s`

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

Encoder lock (must be in run config):
- `encoder_id` (e.g., `resnet18_v1`)
- `weights_source` (`random_init` | `pretrained:<name>`)
- `trainable` (`frozen` | `finetune`)
- `latent_dim` (fixed, default `256`)

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

SlotMap YAML example (smoke-test):
```yaml
slots:
  - slot_id: shelf_A1
    region_world: {center_xyz: [0.90, -1.16, 1.22], size_xyz: [0.18, 0.18, 0.06], yaw: 0.0}
    approach_pose_candidates:
      - {xyz: [0.86, -1.10, 1.32], rpy: [3.14, 0.0, 0.0]}
    place_pose_candidates:
      - {xyz: [0.90, -1.16, 1.22], rpy: [3.14, 0.0, 0.0]}
    allowed_objects: [tray1]
    priority: 1
  - slot_id: shelf_B1
    region_world: {center_xyz: [-0.92, -1.16, 1.22], size_xyz: [0.18, 0.18, 0.06], yaw: 0.0}
    approach_pose_candidates:
      - {xyz: [-0.86, -1.10, 1.32], rpy: [3.14, 0.0, 3.14]}
    place_pose_candidates:
      - {xyz: [-0.92, -1.16, 1.22], rpy: [3.14, 0.0, 3.14]}
    allowed_objects: [tray1]
    priority: 1
```

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
- explicit local dependency bridge to ENPM662 kitchen scene repo (no duplicated large assets)

Tasks:
- [x] scene bridge script validated (`scripts/v5/bridge_kitchen_scene.sh`)
- [x] camera SDF/launch integration verified in scene
- [x] `/v5/cam/*/rgb` + `/camera_info` publishing
- [x] TF check script (`view_frames` / `tf2_echo`) passes
- [x] image health diagnostics (fps, dropped frame, latency)
- [x] rosbag2 record/replay script for camera+state topics
- [x] tray stream extraction (`/tray_tracking/pose_stream_raw` -> `/tray1/pose` GT-only; `/tray_tracking/pose_stream` legacy/disabled-by-default)

Exit criteria:
- 5-min stable camera + state capture
- timestamp latency P95 < 80 ms
- static-scene pose jitter std < 3 mm
- object-id switch rate < 1% (or deterministic selector documented)
- replay reproduces same topic structure
- approx-sync success rate (overhead+side) > 95%
- replay receive latency P95 < 120 ms

### Implementation Log (WP0)
- Clock-domain mixing (`recv_now - header.stamp` across mismatched time bases) caused false latency failures; mitigation: lock latency checks to same ROS clock domain with explicit definition in report output.
- Duplicate launch residue (leftover ROS/Gazebo processes) polluted live checks; mitigation: enforce clean relaunch discipline before WP0 live/replay runs.
- Package-name detection for scene launch was inconsistent across repo layouts; mitigation: resolve launch command through wrapper scripts with dry-run verification.
- `camera_info` bridge parity drifted from RGB topics during integration; mitigation: enforce paired RGB + `camera_info` coverage in WP0 contracts and rosbag record topics.
- Tray topic alignment was inconsistent (canonical `/tray_tracking/pose_stream_raw` vs legacy `/tray_tracking/pose_stream` vs `/tray1/pose`); mitigation: keep explicit extraction path `/tray_tracking/pose_stream_raw` -> `/tray1/pose` for GT-only stability checks.
- `id_switch` source ambiguity (multiple possible streams) reduced reproducibility; mitigation: pin eval source in config/report (`/v5/perception/object_pose_est`) and log source metadata.
- Replay isolation was noisy when live publishers remained active; mitigation: replay checks run with isolated replay path and dedicated replay-latency gate.
- Missing `ROS_LOG_DIR` control made diagnostics hard to trace; mitigation: set per-run log directories under `artifacts/wp0` for repeatable evidence capture.
- Bash vs zsh setup differences caused environment drift; mitigation: standardize wrapper scripts and docs on explicit shell-agnostic invocation plus dry-run command echoing.

## WP1 — L1 layer (phase-aware)
Deliverables:
- `MOVE_PLATE` parser
- SlotMap resolver
- IntentPacket generator
- perception adapter output `/v5/perception/object_pose_est`

Tasks:
- [x] IntentPacket schema
- [x] SlotMap implementation
- [x] Phase-0 GT proxy switch + Phase-1 vision-only switch
- [x] failure path (`unreachable`, `missing object`, `disambiguation required`)

Exit criteria:
- 20 random tasks produce valid intent packets with clean layer boundary

### Implementation Log (WP1)
- Added strict `IntentPacket` schema + validator with explicit L1/L2/L3 boundary guard (forbidden L2/L3 fields are rejected at validation time).
- Landed runtime `SlotMap` loading path from repo config (`hrl_ws/src/hrl_trainer/config/v5_slot_map.yaml`) instead of test-only mock wiring.
- Added perception adapter scaffold for `/v5/perception/object_pose_est` with explicit mode switch:
  - `phase0_gt_proxy`
  - `phase1_vision_only`
- Encoded failure semantics as first-class outcomes:
  - `UNREACHABLE`
  - `MISSING_OBJECT`
  - `TASK_DISAMBIGUATION_REQUIRED`
- Added acceptance harness + runner (`scripts/v5/run_wp1_acceptance.sh`) and summary output contract:
  - `success_count`
  - `fail_count`
  - `fail_reason_breakdown`
- Current acceptance evidence (latest run):
  - 10-task smoke + 20 random acceptance completed
  - success_count=30, fail_count=0, fail_reason_breakdown={}

## WP1.5 — RL pipeline preparation + workspace shaping scaffold
Purpose:
- Build a **rule-based, RL-ready bridge** between WP1 and WP2.
- Do not train final RL policy yet.
- Freeze interfaces and shaping pipeline so WP2 starts from a fixed contract.

Deliverables:
- RL observation contract (policy-visible schema + builder)
- RL action contract (aligned with `SkillCommand`, bounded + safe)
- Workspace exploration pretraining task
- Modular reward/scoring composer (robot-shaping style)
- Rule-based rollout generator (same contracts as future RL)
- Easy/medium/hard curriculum spec
- Reusable rollout artifacts for RL/BC warm-start

Key design rule:
- External execution path stays: `L1 -> Rule-L2 -> L3`
- Internal training prep path is added: `L1 -> obs builder / reward composer / rollout logger`
- Future RL must replace Rule-L2 **without** changing L1/L3 APIs.

Tasks:
- [ ] Freeze observation v1 schema (latent, robot state, stage flag, object pose, target slot/zone)
- [ ] Freeze action v1 schema（`delta_pose` + U-slot 參數 + timing 參數 + bounded guard）
- [ ] Implement action->`SkillCommand` adapter + validation tests
- [ ] Implement workspace zone map + canonical hover/target anchors
- [ ] Implement reward composer modules:
  - progress
  - safety/collision
  - smoothness
  - workspace coverage
  - subgoal/terminal success
- [ ] Implement rule-based rollout generator (with success/fail labels + fail reason)
- [ ] Add deterministic replay + rollout integrity checks
- [ ] Define curriculum YAMLs (`easy`, `medium`, `hard`) with fixed seed sets
- [ ] Export training artifacts (rollout JSON/CSV, reward breakdown, canonical trajectories)

Exit criteria:
- Observation/action contracts are documented + validated by tests
- Rule-based generator can produce reproducible rollouts under all 3 curriculum levels
- Reward breakdown is logged per step and per episode
- Rollout artifacts are reusable for WP2 warm-start (RL/BC)
- No L1/L3 API changes required to swap in RL-L2

Implementation notes (WP1.5 pitfalls to avoid):
- Avoid moving target contracts (freeze schema before large-scale rollout generation)
- Keep L2 learning surface clean; do not leak L3 internals into policy API
- Keep shaping modular/config-driven (weight changes without code edits)
- Treat WP1.5 as engineering prep, not a side research detour

## WP2 — RL-L2 slot（參數化中階連續策略）

> 依賴關係（硬規則）
> - WP2 不可先於 WP3-Lite。先完成 L3 最小可用執行/安全底座，再做 RL-L2 訓練。
> - Runtime gate 必須先 PASS：`scripts/v5/run_wp1_5_runtime_parity_check.sh --mode both`

Deliverables:
- Rule-L2 v0 baseline（可重跑、可對照）
- RL-L2 v2（參數化中階連續策略，非 trajectory generator）
- RL-L2 / Rule-L2 benchmark 對照結果（easy/medium）

WP2 固定頻率契約（已拍板）：
- L2：10–20 Hz（事件觸發重算）
- L3：100–200 Hz（deterministic control + safety shield）
- L2/L3 互動：stale timeout + interpolation + predictive clamp + fail-safe fallback

Tasks（具體化）:
- [ ] M1: Rule-L2 v0 封裝（沿用既有 controller 思路）
  - 只輸出 `SkillCommand`，不得輸出 trajectory chunk / spline / JointTrajectory
  - 產生 rule rollout artifact（easy/medium）
- [ ] M2: RL-L2 action schema v2（參數化擴展）
  - 在不破壞 L3 邊界下，增加可學習連續控制意圖（pose/U-slot/timing/safety-hint）
  - 每步輸出 reward breakdown 與 action telemetry
- [ ] M3: Reward composer v1（Sparse + PBRS + safety + smooth）
  - 支援 config-driven 權重調整（不改程式碼即可改權重）
- [ ] M4: Curriculum + warm-start
  - Stage A/B/C（workspace exploration -> atomic skills -> task composition）
  - Rule rollout 生成 BC warm-start dataset
- [ ] M5: RL baseline 訓練與對照
  - SAC baseline（fixed seeds）
  - ablation：no shaping vs heuristic dense vs PBRS

Exit criteria（DoD）:
- Interface success:
  - RL-L2 可替換 Rule-L2，且不改 L1/L3 API
  - L2 輸出嚴格通過 `SkillCommand` boundary validator
- Runtime success:
  - parity gate PASS（manual + auto）
- Training success:
  - easy benchmark 上，RL-L2 success rate 不低於 Rule-L2
  - medium benchmark 上至少改善一項 KPI（success/collision/time）
  - reward breakdown、telemetry、rollout logs 可重現

## WP3 — L3 execution + safety（WP2 先決 Lite + 完整版）
Deliverables:
- deterministic executor
- safety shield
- structured intervention/failure logs

Tasks:
- [ ] WP3-Lite（WP2 前置）
  - projection to safe command space（先可用）
  - stale command handling（hold/interpolate/clamp/timeout stop）
  - basic intervention triggers（distance/tilt/ttc）
- [ ] WP3-Full（WP2 並行收斂）
  - intervention triggers + recovery policy 完整化
  - thresholds configurable in yaml + triggered metric logged
  - enum-based structured logs + replay-friendly diagnostics

Exit criteria:
- no unsafe command reaches controller in stress scenarios
- L2 command loss/stale 情境下，L3 可穩定降級與安全停機

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
- image latent(s) from observation pipeline（overhead/side）
- robot state（joint state、ee pose/twist、controller state）
- stage flag / current skill mode
- perception estimate（`/v5/perception/object_pose_est`）
- tray 相對位姿特徵（EE->tray frame 相對平移/yaw）
- U-slot 幾何關聯特徵（estimated slot-edge distance、clearance proxy）
- command staleness 與 safety context（intervention recent flag）

Action space (learned fields in SkillCommand):
- `skill_mode` switch policy（`APPROACH -> INSERT_SUPPORT -> LIFT_CARRY -> PLACE -> WITHDRAW`）
- `delta_pose` / `ee_target_pose` selection
- U-slot insertion params（`insert_depth`, `lateral_alignment`, `vertical_clearance`, `entry_yaw`）
- timing params（`approach_speed_scale`, `lift_profile_id`, `contact_settle_time`）
- guard hint selection within bounded range（含 `fragility_mode_hint`）

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

## 10.2 Stagewise RL training policy

Stage A — Workspace exploration
- objective: learn safe hover, zone transition, and clearance-aware motion
- task shape: exploration-focused; no object manipulation success requirement

Stage B — Atomic skill training
- objective: stabilize `APPROACH / GRASP / PLACE / RETREAT`
- task shape: skill-level training with shaping rewards and strict SkillCommand boundaries

Stage C — Task composition
- objective: learn `MOVE_PLATE(A,B)` end-to-end under frozen contracts
- task shape: full pipeline RL-L2 training with curriculum + failure recovery

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

### Phase-1 minimum acceptance gates
- `success_rate >= 0.70` on easy eval set
- `collision_rate <= 0.05`
- `timeout_rate <= 0.10`
- `safety_intervention_rate` below configured threshold
- fixed config + seed results reproducible within agreed tolerance

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


## 13) Practical risk-control refinements

### R1. Perception adapter is the primary risk
Observed risk:
- identity switch / confidence oscillation / pose jitter can destabilize L1->L2->L3.

Execution policy:
1. Phase-0: GT proxy adapter for chain validation only.
2. Phase-1a: simple estimator (bbox/coarse pose first, no over-optimized 6D requirement).
3. Phase-1b: replace estimator with VLM/VLA adapter under the same `ObjectPoseArray` contract.

Rule:
- keep perception adapter pluggable; never couple policy code to one estimator implementation.

### R2. Camera IO + rosbag throughput/latency
Observed risk:
- WSL2 + multi-RGB recording can break replay latency and sync constraints.

Execution policy:
- run throughput stress test before locking final camera spec.
- if replay latency budget is violated, downgrade in this order:
  1) lower FPS
  2) lower resolution
  3) switch to `CompressedImage`
- every downgrade must update hard spec and benchmark config in versioned files.

### R3. Safety threshold tuning will be iterative
Observed risk:
- trigger thresholds depend on distance backend/frame choice and latency noise.

Execution policy:
- all thresholds remain yaml-configurable.
- each intervention log must include `trigger_metric_value` and `active_threshold`.
- tune from conservative->relaxed only when safety KPIs remain stable.

### R4. RL-L2 drift into reward-tuning game
Observed risk:
- RL may exploit reward/termination artifacts instead of improving architecture quality.

Execution policy:
- Rule-L2 baseline is mandatory and continuously evaluated.
- single-variable change per experiment round (reward OR termination OR curriculum).
- if RL-L2 fails to beat baseline on agreed KPI, treat as non-progress and rollback.

## 14) Immediate Next Actions

### Execution now
1. Finish WP1 minimal (`IntentPacket`, `SlotMap`, `/v5/perception/object_pose_est` adapter).
2. Freeze WP1.5 observation/action v1 schema.
3. Implement reward composer + rule rollout generator.
4. Define workspace zone map + canonical hover anchors.
5. Generate first exploration rollouts and verify replay reproducibility.

### Deferred but queued
- camera hardening refinements
- perception estimator swap experiments
- Phase-1 visual encoder integration

---

## 15) File/Code Placement

- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/intent_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/perception_adapter.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/skill_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/control_layer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/pipeline.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/rl_observation.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/rl_action_adapter.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/reward_composer.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/workspace_exploration.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/rule_rollout_generator.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/v5/curriculum.py`
- `hrl_ws/src/hrl_trainer/config/v5_kitchen_*.yaml`
- `docs/V5_EXPERIMENT_LOG.md`

---

## 16) Acceptance Gate

- [ ] Phase split accepted (Phase-0 GT / Phase-1 Vision)
- [ ] L2 strict output boundary accepted
- [ ] camera contract accepted
- [ ] observation pipeline accepted
- [ ] SlotMap spec accepted
- [ ] RL prep scaffold accepted (obs/action contract + reward composer + rollout artifact path)
- [ ] safety enums/logging accepted
- [ ] timeline accepted

Once these are accepted, convert directly into final proposal/execution handbook.
