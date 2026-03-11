# RL_brain_trainer (V5 Active / V4 Legacy)

## 中文版

本 repo 目前開發焦點是 **V5（manipulation）**，V4（sim2d）保留為可重現基線：
以「可驗證、可重跑、可逐步升級」為目標，V5 持續推進手臂場景三層架構，V4 作為歷史主線與對照實驗基底。

### 1) 架構總覽（L1 / L2 / L3）
- **L1（語意/全域意圖）**：產生任務意圖與約束，不直接輸出控制量。
- **L2（局部規劃 / policy）**：根據狀態與目標產生局部決策（可含記憶/殘差策略）。
- **L3（deterministic follower）**：把 L2 指令轉成可執行控制，負責可追蹤性與穩定性。

核心契約：把「理解」和「技能」分開，把「技能」和「精度」分開。  
介面細節：`docs/V4_INTERFACE_SPEC.md`

### 2) V4 關鍵機制
**控制約束（Control Constraints）**
- `v ∈ [-1.2, 1.2]`（允許倒車）
- `ω ∈ [-2π, 2π]`（±360 deg/s）

**Reward（目前實作）**
- 距離/進度相關項
- 控制 effort 懲罰
- 成功獎勵（success bonus）
- 碰撞懲罰（collision penalty）

備註：目前 V4 訓練路徑沒有獨立命名為 backtracking loss 的顯式項。

### 3) 實驗方法（V4）
**代表性訓練設定**
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

**執行範例**
```bash
cd hrl_ws
uv sync
source .venv/bin/activate

python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

### 4) 最新結果（V4）
- No-obstacle MVP：success rate **95.83%**（115/120）
- Complex + Velocity Constraints：success rate **95.56%**（172/180）, collisions **8**, timeout **0**

### 5) 參考文獻
- 索引：`docs/literature/INDEX.md`
- 閱讀路徑：`docs/literature/PIPELINE_GUIDE.md`
- 關聯文獻：RoBridge / Mem-α / Memory-R1 / RLBenchNet / LAC

### 6) 目錄重點
- 程式：
  - `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/env.py`
  - `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py`
- 文件：
  - `docs/V4_INTERFACE_SPEC.md`
  - `docs/archive/v3_legacy/V3_VELOCITY_PATCH_REPORT.md`
  - `docs/archive/v3_legacy/V3_GUIDE.md`
  - `docs/archive/v3_legacy/V3_PROGRESS_BOARD.md`

### 7) Branch 策略
- `main`：V4 主線
- `
下一階段：**V5 模擬手臂場景（manipulation）**。

---

## English Version

This repository is now **V5-active (manipulation)**, with V4 (sim2d) kept as a legacy/reproducible baseline.
The goal is to keep a verifiable, reproducible hierarchy while advancing V5 robot-arm manipulation and preserving V4 as a reference line.

### V5 Status Snapshot
- WP0 runtime for V5 kitchen work depends on the ENPM662 kitchen scene repo.
- Integration/launch/healthcheck workflow is documented in `docs/V5_SCENE_DEPENDENCY.md` and `docs/wp0_run.md`.
- WP0 validation status (2026-03-09): overall `PASS` with `6/6` sections passing, including rosbag replay gate.
- WP1 foundation status (latest): IntentPacket/SlotMap/perception-adapter scaffold + acceptance harness landed (see `docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md`).

### 1) Architecture Overview (L1 / L2 / L3)
- **L1 (semantic/global intent):** outputs task intent and constraints, not low-level controls.
- **L2 (local planning/policy):** generates local decisions from state/goal (optionally memory/residual policy).
- **L3 (deterministic follower):** converts L2 outputs into executable controls with tracking stability.

Core contract: separate understanding from skill, and skill from precision.  
Interface details: `docs/V4_INTERFACE_SPEC.md`

### 2) V4 Key Mechanisms
**Control constraints**
- `v ∈ [-1.2, 1.2]` (reverse enabled)
- `ω ∈ [-2π, 2π]` (±360 deg/s)

**Reward terms (current implementation)**
- distance/progress-related term
- control effort penalty
- success bonus
- collision penalty

Note: there is no explicitly named “backtracking loss” term in the current online_v4 path.

### 3) Experimental Method (V4)
**Representative configs**
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

**Run example**
```bash
cd hrl_ws
uv sync
source .venv/bin/activate

python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

### 4) Latest Results (V4)
- No-obstacle MVP: success rate **95.83%** (115/120)
- Complex + velocity constraints: success rate **95.56%** (172/180), collisions **8**, timeout **0**

### 5) References
- Index: `docs/literature/INDEX.md`
- Reading path: `docs/literature/PIPELINE_GUIDE.md`
- Key references: RoBridge / Mem-α / Memory-R1 / RLBenchNet / LAC

### 6) Important Files
- Code:
  - `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/env.py`
  - `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v4.py`
- Docs:
  - `docs/V4_INTERFACE_SPEC.md`
  - `docs/archive/v3_legacy/V3_VELOCITY_PATCH_REPORT.md`
  - `docs/archive/v3_legacy/V3_GUIDE.md`
  - `docs/archive/v3_legacy/V3_PROGRESS_BOARD.md`

### 7) Branch Strategy
- `main`: V4 mainline (only)

V2/V3 materials are archived under `docs/archive/v3_legacy/` and are no longer part of the active path.

Next stage: **V5 robot-arm simulation (manipulation)**.
