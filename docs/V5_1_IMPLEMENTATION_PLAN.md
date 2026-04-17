# V5.1 實作計劃書（正式版）

- **版本**: v1.0 (正式版)
- **日期**: 2026-03-27
- **適用範圍**: `RL_brain_trainer` V5.x -> V5.1 落地
- **核心決議來源**:
  - Jerry 最新三層決策：**L1 perception+decision / L2 joint-space RL / L3 deterministic execution+safety**
  - `reports/V5_1_JR_RESEARCH_ALIGNMENT.md`
  - 現有 V5 文件（`V5_KITCHEN_IMPLEMENTATION_PLAN.md`, `V5_KITCHEN_THREE_LAYER_PLAN_DRAFT.md`, `V5_DESIGN_PHILOSOPHY.md`）

---

## 0) 目標與非目標

### 目標
1. 把 V5 現有可重用資產收斂成 **V5.1 單一可執行方案**，直接可拆票實作。
2. 凍結 L2/L3 核心契約，避免後續接口漂移。
3. 以 **SAC baseline + robot shaping curriculum** 建立可重跑、可比較、可回滾的訓練路徑。

### 非目標（本版不做）
1. 不做端到端 vision-to-action 一體化策略。
2. 不在本版導入 world model / diffusion policy / 大型 VLA 替代 baseline。
3. 不把 L3 安全邏輯下放給 RL policy。

---

## 1) 架構決議（保留 / 廢止 / 重構）

> 原則：**保留三層，重構 L2/L3 契約，廢止與 V5.1 衝突輸出形態。**

### 1.1 保留（Keep）

1. **三層邊界**（L1/L2/L3）
   - 來源：`V5_DESIGN_PHILOSOPHY.md`
   - 保留原因：可診斷、可分層測試、與目前實作資產一致。

2. **L1 IntentPacket + SlotMap 任務語義流**
   - 來源：`V5_KITCHEN_IMPLEMENTATION_PLAN.md`
   - 保留原因：高層任務輸入已可用，與 L2 解耦良好。

3. **L3 deterministic-first + safety shield**
   - 來源：`V5_KITCHEN_IMPLEMENTATION_PLAN.md`, `V4_INTERFACE_SPEC.md`
   - 保留原因：安全可控與可審計。

4. **WP0/WP1/WP1.5 既有基礎腳本與檢查流程**
   - 來源：`scripts/v5/*`
   - 保留原因：已有 runtime parity、acceptance、healthcheck 骨架。

### 1.2 廢止（Deprecate）

1. **L2 直接輸出 trajectory chunk / spline / JointTrajectory 的語義**
   - 廢止原因：違反 L2/L3 邊界；L3 才是唯一 executable owner。

2. **absolute joint target 當作預設 action space**
   - 廢止原因：實機風險與抖動成本高；V5.1 以 bounded `delta_q` 為基準。

3. **L1/L2/L3 同頻更新假設**
   - 廢止原因：與實際控制架構不符；改為多速率（L1低/L2中/L3高）。

### 1.3 重構（Refactor）

1. **L2 action contract** -> 統一為 joint-space `delta_q`（bounded）
2. **L2 observation contract** -> 固定低維優先（proprioception + task phase + error terms）
3. **L3 safety contract** -> clamp/rate/projection/watchdog/intervention log 五件套標準化
4. **實驗流程** -> 三個優先實驗固定成 baseline gate（A/B/C）
5. **票務映射** -> 以可交付 artifact 定義 DoD（不是口頭進度）

---

## 2) L2 Observation / Action Contract（V5.1 凍結版）

## 2.1 Observation contract（policy-visible）

`obs_t` 必含欄位（v1）：

1. `q`：關節角（可控關節）
2. `dq`：關節角速度
3. `ee_pose_error`：相對當前 subgoal 的位置/姿態誤差（建議 6D）
4. `prev_action`：前一步 action（穩定學習）
5. `task_phase_id`：L1 輸出的階段 ID（one-hot 或 embedding）
6. `safety_context`：最近 N 步 safety intervention flag / clip ratio（低維摘要）

可選欄位（v1.1+）：
- `object_pose_est`（L1 perception 輸出，非 GT raw）
- `goal_tolerance_profile_id`

禁止欄位：
- policy 直接讀取 GT raw stream（僅 reward/eval 可用）
- L3 內部控制器狀態細節（超出契約）

## 2.2 Action contract（joint-space, bounded delta_q）

- action 向量：`a_t ∈ R^n`, `n = #controlled_joints`
- 網路輸出：`u_t = tanh(raw_t)`
- 實際命令：`delta_q_t = u_t * delta_q_max_per_joint`
- 命令前處理：
  - `delta_q_t = clip(delta_q_t, -delta_q_max, +delta_q_max)`
  - `q_cmd = q_t + delta_q_t`
- 最終下發前，仍必經 L3 projection/safety filter。

## 2.3 建議起始邊界（可配置）

- `delta_q_max_per_step`: 每關節額定步進上限的 5%~10%
- `joint_limit_margin`: 5 deg
- `max_vel_ratio`: 0.3
- `max_acc_ratio`: 0.3
- `max_jerk_ratio`: 0.2

---

## 3) SAC Baseline 實作規格（V5.1 default）

## 3.1 模型結構

- **Actor**: Gaussian policy, MLP `[256, 256]`, ReLU
- **Critic**: Twin Q-networks, each MLP `[256, 256]`, ReLU
- **Target critic**: Polyak update (`tau=0.005`)
- **Entropy**: automatic temperature tuning

## 3.2 超參數（初版固定）

```yaml
algo: SAC
seed: [0, 1, 2]
gamma: 0.99
tau: 0.005
lr_actor: 3e-4
lr_critic: 3e-4
lr_alpha: 3e-4
batch_size: 256
replay_size: 1000000
start_steps: 10000
updates_per_step: 1
max_episode_steps: 200
eval_every_steps: 10000
eval_episodes: 20
checkpoint_every_steps: 50000
total_steps: 1000000
```

## 3.3 訓練循環（硬流程）

1. reset env，收集 `obs_0`
2. `t < start_steps` 用 random action；之後用 actor sampling
3. 取得 `delta_q` -> 經 L3 safety -> env step
4. 寫入 replay：`(obs, action, reward, next_obs, done, safety_tags)`
5. 每步做 `updates_per_step` 次 SAC update（actor/critic/alpha）
6. 每 `eval_every_steps` 做 deterministic eval 並記錄 KPI
7. 每 `checkpoint_every_steps` 存權重 + config hash + git commit hash

## 3.4 必記錄欄位

- RL core: `episode_return`, `success_rate`, `critic_loss`, `actor_loss`, `alpha`
- safety: `intervention_rate`, `clip_ratio`, `hard_stop_count`
- motion quality: `jerk_mean`, `action_l2`, `limit_proximity_ratio`
- reproducibility: `seed`, `config_hash`, `commit_hash`

---

## 4) Robot Shaping Curriculum（漸進開放）

> 固定路徑：**鎖關節 / 限幅 / 小 workspace -> 逐步開放**

## Stage S0 — Safety Bring-up（無學習或固定 policy）
- 設定：鎖高風險關節、workspace=40~60%、低速低加速度
- Gate：連續 N episode `collision=0` 且 `hard_limit_hit=0`

## Stage S1 — 低自由度 RL（2~3 joints）
- 設定：平面主關節 + 嚴格 `delta_q_max`
- Gate：`success_rate >= 0.70`，`intervention_rate <= 0.05`

## Stage S2 — 擴關節 + 擴 workspace
- 設定：4~5 joints，workspace=70~80%
- Gate：成功率相對 S1 降幅 <= 10%，碰撞率不惡化

## Stage S3 — 全關節任務化
- 設定：全關節、起終點隨機化、目標位姿擾動
- Gate：跨 seed 穩定（std <= 10%），長尾失敗可分類

## Stage S4 — 視覺增強（可選）
- 前提：S3 gate 已通過
- 設定：視覺特徵 late-fusion，不取代低維核心
- Gate：性能不退化且方差可控

---

## 5) L3 安全與執行契約（Deterministic + Auditable）

## 5.1 執行順序（固定）

`L2 delta_q_cmd -> clamp -> rate limit -> projection -> watchdog -> controller`

## 5.2 五大安全機制

1. **Clamp**
   - 關節、速度、加速度、jerk 硬裁切
2. **Rate limiter**
   - 防 command 抖動與突變
3. **Projection**
   - 將不可行命令投影到安全可行域（關節界限/工作空間/避碰近似）
4. **Watchdog**
   - stale command timeout、控制回授異常、通訊中斷保護
5. **Intervention log**
   - 結構化記錄 `type/reason/metric/threshold/timestamp/recovery_action`

## 5.3 Intervention enum（v1）

- `NONE`
- `CLAMP_ONLY`
- `RATE_LIMITED`
- `PROJECTION_CLAMP`
- `SLOWDOWN`
- `HALT`
- `RETREAT`

## 5.4 FailReason enum（v1）

- `IK_FAIL`
- `LIMIT_NEAR_VIOLATION`
- `COLLISION_RISK`
- `CONTACT_DETECTED`
- `COMMAND_STALE`
- `TIMEOUT`
- `CONTROLLER_FAULT`

---

## 6) 實驗矩陣與成功門檻（3 個優先實驗）

## Exp-A：Low-dim SAC 收斂基線（必跑）

- 設定：S1 curriculum，3 seeds，1M steps
- 成功門檻：
  1. `success_rate >= 0.80`
  2. `intervention_rate <= 0.05`
  3. 跨 seed `std <= 0.10`

## Exp-B：`delta_q` vs `absolute_q` 對照（必跑）

- 設定：相同網路/seed/任務，僅 action 定義不同
- 成功門檻（delta_q 相對 absolute_q）：
  1. `jerk_mean` 降低 >= 20%
  2. `intervention_rate` 降低 >= 30%
  3. 成功率不低於對照組

## Exp-C：Curriculum vs Full-open（必跑）

- 設定：
  - C1: S1->S2->S3 漸進
  - C2: 全關節全workspace from scratch
- 成功門檻（C1 相對 C2）：
  1. 達 70% 成功率所需 samples 減少 >= 25%
  2. 碰撞率降低 >= 30%
  3. 訓練崩潰次數顯著較少

---

## 7) 風險、回滾、里程碑與票務映射

## 7.1 主要風險與應對

1. **Q-value 發散**
   - 應對：降 learning rate、reward normalization、縮小 action scale
2. **安全裁切過高（policy 常撞邊界）**
   - 應對：加 boundary penalty、收緊 curriculum gate
3. **課程切換退化**
   - 應對：增加中間 stage、混合 replay、延長適應期
4. **sim2real 落差過大**
   - 應對：delay/randomization/system ID（列為下一里程碑）

## 7.2 回滾策略（硬規則）

- 若任一版本出現以下情況，立即回滾到上一穩定 checkpoint：
  1. `collision_rate` 連續 3 次評估超閾值
  2. `intervention_rate` 激增且無收斂趨勢
  3. Q 指標持續發散（非短暫波動）

回滾資產要求：
- 最近穩定模型權重
- 對應 config YAML
- 對應 commit hash + report JSON

## 7.3 里程碑（M）

- **M1**: 凍結 L2/L3 契約與 config schema（DoD: schema tests PASS）
- **M2**: SAC baseline 訓練循環可重跑（DoD: 3 seeds 可產生完整報告）
- **M3**: Curriculum S0~S3 gate 跑通（DoD: stage gate artifacts 完整）
- **M4**: Exp-A/B/C 完成並出比較報告（DoD: 門檻判定明確）

## 7.4 票務映射（建議）

- **Ticket 2.1**：L2 obs/action schema 凍結 + validator
- **Ticket 2.2**：SAC baseline trainer（actor/critic/replay/update/eval/checkpoint）
- **Ticket 2.3**：L3 safety pipeline（clamp/rate/projection/watchdog/log）
- **Ticket 2.4**：curriculum engine（S0~S4 + gate checker）
- **Ticket 2.5**：experiment harness（Exp-A/B/C + report generator）
- **Ticket 2.6**：rollback tooling（checkpoint selector + regression guard）

每票都需交付：程式、測試、run command、artifact 路徑、DoD 結果。

---

## 8) 一鍵執行 / 重跑命令草案

> 目標：讓 V5.1 可「同一入口重跑」。以下為草案命令（以 repo 現有腳本風格設計）。

## 8.1 前置健康檢查

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
./scripts/v5/run_wp0_healthcheck.sh
./scripts/v5/run_wp1_5_runtime_parity_check.sh --mode both
```

## 8.2 V5.1 Baseline 全流程（草案）

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/v5/run_v5_1_baseline.sh \
  --config hrl_ws/src/hrl_trainer/config/v5_1_sac_baseline.yaml \
  --seeds 0,1,2 \
  --stages S0,S1,S2,S3
```

## 8.3 三個優先實驗（草案）

```bash
# Exp-A
bash scripts/v5/run_v5_1_experiment.sh --exp A --config hrl_ws/src/hrl_trainer/config/v5_1_exp_a.yaml

# Exp-B
bash scripts/v5/run_v5_1_experiment.sh --exp B --config hrl_ws/src/hrl_trainer/config/v5_1_exp_b.yaml

# Exp-C
bash scripts/v5/run_v5_1_experiment.sh --exp C --config hrl_ws/src/hrl_trainer/config/v5_1_exp_c.yaml
```

## 8.4 一鍵重跑 + 匯總（草案）

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
bash scripts/v5/rerun_v5_1_all.sh
```

預期輸出：
- `artifacts/reports/v5_1/exp_a_summary.json`
- `artifacts/reports/v5_1/exp_b_summary.json`
- `artifacts/reports/v5_1/exp_c_summary.json`
- `artifacts/reports/v5_1/v5_1_gate_decision.md`

---

## 9) 驗證方式（本計劃書可直接轉實作）

1. **文檔驗證**：本文件 8 個必含章節已完整覆蓋。
2. **契約驗證**：L2/L3 contract 有明確欄位、邊界、禁止項。
3. **訓練驗證**：SAC baseline 含模型、超參、訓練循環、記錄欄位。
4. **實驗驗證**：3 個優先實驗均有可量化門檻。
5. **工程驗證**：提供票務映射 + 一鍵執行草案，可直接開工拆票。

---

## 附註：與現有 V5 文檔的銜接策略

- 本文件不推倒重來；採 **兼容升級**：
  - 延續 `V5_DESIGN_PHILOSOPHY` 的三層哲學
  - 延續 `V5_KITCHEN_IMPLEMENTATION_PLAN` 的 pipeline 與 safety 思路
  - 將 `V5_KITCHEN_THREE_LAYER_PLAN_DRAFT` 的草案內容正式化為可交付規格
- 任何新實作若違反本文件第 1/2/5 節，視為偏離 V5.1，需先提變更審查。
