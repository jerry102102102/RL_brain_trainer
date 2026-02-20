# V4 Interface Spec (L1/L2/L3 + Memory + Safety)

> 目的：把 V4 的層間契約、state schema、memory 操作、以及 safety shield 插拔點明確化，避免實作時語義漂移。

## 1) Layer Contract（不變）

- **L1（Intent/Constraint Representation）**
  - 職責：語意理解、全域約束表徵
  - 禁止：直接輸出控制量
- **L2（Policy + Memory Controller）**
  - 職責：局部規劃/技能決策，輸出可執行局部目標
  - 位置：V4 唯一學習主體
- **L3（Deterministic Follower + Safety Shield）**
  - 職責：追蹤控制、穩定與安全兜底
  - 性質：deterministic，可插拔 safety shield

---

## 2) I/O Contract

### 2.1 L1 -> L2

`IntentPacket`（目前實作最小集合）
```yaml
subgoal_xy: [float, float]
speed_hint: float
# 後續可擴充
# feasible_set_descriptor: ...
# risk_hints: ...
```

### 2.2 L2 -> L3

`DesiredVO`
```yaml
v_target: float
omega_target: float
```

### 2.3 L3 -> Env

`ControlAction`
```yaml
a_lin: float   # normalized in [-1, 1]
a_ang: float   # normalized in [-1, 1]
```

---

## 3) L2 State Schema（V4-MVP 現況）

目前 policy 單步輸入為 15 維：

1. env obs（10 維）
2. subgoal delta（2 維：dx, dy）
3. speed_hint（1 維）
4. retrieved memory vector（2 維，無檢索時為 0）

> recurrent 模式：`seq_len x 15`（例如 10x15）

---

## 4) Memory Contract

## 4.1 Memory Bank（三層目標結構）

- `core`：低頻、長期、硬約束/物理常識
- `semantic`：中頻、可泛化策略原型
- `episodic`：高頻、近期成功/失敗片段

## 4.2 Memory Item Schema（建議統一）

```yaml
type: core | semantic | episodic
key: vector[d]
value: vector[k]
meta:
  created_step: int
  last_used_step: int
  success_tag: bool
  collision_tag: bool
  risk_level: float
quality: float
cost: float
```

## 4.3 Memory Manager Action Space（V4 目標）

```yaml
op in {ADD, UPDATE, DELETE, NOOP, RETRIEVE}
```

> V4-MVP 現況：已做 retrieval + quality/eviction；`ADD/UPDATE/DELETE/NOOP` 的 RL manager 仍在落地中。

---

## 5) L3 Safety Shield Contract（插拔點）

`L2 desired_vo` -> `ShieldProjection` -> `Follower`

1. 先接收 L2 `desired_vo`
2. 經 safety shield（CBF/CLF-QP）做可行域投影
3. 若 QP infeasible 或風險超閾值，觸發 backup controller
4. 交給 deterministic follower 轉成最終 action

### 5.1 必記錄指標

- safety violation count
- QP infeasible count
- backup trigger rate
- projected action ratio（被修正比例）

---

## 6) 啟用率 / 計算頻率（回答：三層不應每步同頻）

### 6.1 目前簡化狀態（研究階段可接受）

- L1 / L2 / L3 幾乎同頻更新（每步）

### 6.2 實務建議（多速率）

- **L1：低頻**（例如 1~2 Hz，事件觸發重算）
- **L2：中頻**（例如 5~20 Hz，局部規劃與記憶決策）
- **L3：高頻**（例如 50~200 Hz，穩定追蹤與安全過濾）

### 6.3 事件觸發重算條件（建議）

L1 重算：
- 任務切換/新語意約束/全域路徑失效

L2 重算：
- subgoal 到達、局部可行域改變、記憶檢索分數低於閾值

L3 高頻持續：
- 永遠在線（控制回路）

> 結論：你的直覺正確。在複雜系統中三層啟用率通常差 1~2 個數量級。

---

## 7) 實作檢查清單

- [ ] L1 沒有直接輸出 action
- [ ] L2 輸出介面固定為 `desired_vo`
- [ ] L3 可以在不改 L2 的情況下插拔 safety shield
- [ ] memory manager 操作可審計（operation log）
- [ ] 報告包含啟用率統計（L1/L2/L3 call counts / Hz）
