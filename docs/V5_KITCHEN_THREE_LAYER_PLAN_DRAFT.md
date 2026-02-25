# V5 Kitchen Robot — Three-Layer RL Structure Draft

> 目的：把 RL brainer 的三層結構落地到 kitchen robot 場景，任務指令只保留高層自然語意：
> 「把盤子從 A 搬到 B」，其餘感知/規劃/控制由三層架構完成。

## 0) 任務定義（固定）
- **使用者輸入**：`MOVE_PLATE(source_slot, target_slot)`
- **系統輸出**：
  1) 任務完成（盤子放置到目標槽位，姿態/偏差在容許範圍）
  2) 或失敗原因（不可達、碰撞風險、抓取失敗、超時）

---

## 1) 三層對應（Kitchen 版）

### L1 — Perception + High-level Task Reasoning（腦）
**責任**
- 解析任務語意（A→B）
- 感知場景狀態（盤子/手臂/障礙）
- 輸出約束與子目標序列，不直接輸出關節控制

**輸入**
- 相機/場景位姿流（tray tracking, arm state, world state）
- 任務指令

**輸出（IntentPacket）**
- `object_id=tray1`
- `pick_pose`, `place_pose`
- `constraints`（避碰、安全高度、速度上限、時間預算）
- `subtask_graph`（approach→grasp→lift→transport→place→retreat）

---

### L2 — Skill Policy / Local Planner（肌肉記憶）
**責任**
- 把 L1 子任務轉成局部可執行 skill 指令
- 根據局部狀態做短視窗決策（抓取角度、過渡 waypoint、補償）

**輸入**
- L1 IntentPacket
- 當前 robot/joint/EE 狀態
- 物件位姿（tray pose）

**輸出（SkillCommand）**
- `ee_target_pose` 或 `joint_trajectory_chunk`
- `gripper_cmd`（open/close）
- `mode`（approach/grasp/carry/place）

---

### L3 — Deterministic Controller + Safety Shield（穩定層）
**責任**
- 將 L2 指令轉為高頻可追蹤控制
- 碰撞/關節限制/奇異位形保護
- 失配時觸發 fallback（減速/停止/回退）

**輸入**
- SkillCommand
- 關節反饋/控制器狀態

**輸出**
- `/arm_controller/joint_trajectory`（或等價控制介面）
- `execution_status`（success/fail + reason)

---

## 2) 與現有進度的銜接點

已具備：
- 場景可啟動（headless）
- 手臂狀態 topic 可讀
- arm controller 可接收最小 trajectory
- tray 追蹤 stream 已打通（world pose stream）

待補：
- 將 `tray_tracking/pose_stream` 收斂為「單一 tray1 pose topic」
- 明確定義 frame 與時間同步規則
- L1/L2/L3 介面 proto（json/yaml/msg）固定化

---

## 3) 實驗設計（用來驗證三層框架）

### Stage A — Integration Baseline（非學習）
- L1/L2 先用規則器，L3 用現有控制器
- 驗證端到端任務可以跑通（A→B）
- 目標：建立 deterministic 上限基線

### Stage B — L2 RL 啟動（主驗證）
- 固定 L1/L3，僅 L2 policy 學習
- 比較：
  - Rule L2 vs RL L2
  - 無 memory vs memory-assisted
- 指標：success、collision、completion time、trajectory smoothness

### Stage C — L1 感知/任務決策上移
- 將場景複雜度提高（遮擋、目標變更、路徑受阻）
- 驗證 L1 重規劃可否提高韌性

---

## 4) 指標與驗收

**主指標**
- Task Success Rate
- Collision Rate
- Mean Completion Time
- Replan Count（L1/L2）
- Safety Intervention Rate（L3）

**驗收門檻（草案）**
- Baseline（rule）：可重現完成率 > 80%
- RL-L2：相對 baseline 成功率提升或時間下降（至少一項）
- 失敗可分類且可重跑（非黑箱）

---

## 5) 最小可執行路線（MVP）
1. 固化單一 tray1 pose topic（SSOT）
2. 定義 L1->L2、L2->L3 I/O schema
3. 建立 rule-based A→B 任務執行器（端到端）
4. 接入 RL-L2 取代 rule-L2
5. 跑 3 組難度（easy/medium/hard）做對照

---

## 6) 風險與對策
- **感知抖動**：加濾波 + 時戳對齊 + frame 鎖定
- **控制震盪**：L3 增加速度/加速度限制與安全投影
- **策略過擬合**：domain randomization + hold-out 任務集
- **實驗不可追溯**：每輪固定輸出 config/hash/metrics/artifacts

---

## 7) 你接下來寫計劃書可直接用的章節骨架
1) Problem Statement
2) Three-layer Architecture Mapping
3) System Interface Contract
4) Experimental Protocol
5) Metrics & Acceptance Criteria
6) Risk Mitigation
7) Timeline & Deliverables
