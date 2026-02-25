# Literature Index for RL_brain_trainer (V4)

> 目的：把目前已收錄 PDF 依「架構環節」分門別類，方便快速找到可用論文。
> 範圍：`/home/jerry/.openclaw/media/inbound/*.pdf` 中與研究有關者。

## A. L1（高層語意/認知規劃）

### A1. Cognition → execution 分層架構
- `160f9154-ea6e-4e4c-8952-bd4fe027235a.pdf`
  - Title: **RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation**
  - 用途：可直接對照我們的 L1/L2/L3 分層合理性與接口設計。
  - 優先級：⭐⭐⭐⭐⭐

## B. L2（局部規劃 + 記憶 + 受限行動）

### B1. Memory-based constrained action selection
- `dee63f08-7672-497a-8d7c-3840e9166c60.pdf`
  - First-page topic: **Using Memory-Based Learning to Solve Tasks with State-Action Constraints**
  - 用途：定義 memory 檢索/門控/替換策略，避免 L2 盲目修正。
  - 優先級：⭐⭐⭐⭐⭐

### B2. Control/optimization under constraints（待精讀）
- `d0072133-43f3-4a58-bf54-1f57e7511e1a.pdf`
  - First-page source: *International Journal of Optimization and Control*
  - 用途：補 L2 constrained planner 的數學化與優化框架。
  - 優先級：⭐⭐⭐

### B3. Memory policy learning（RL 記憶建構）
- `a4a27341-4a4b-4b09-867a-c555ea5021e6.pdf`
  - Title: **Mem-α: Learning Memory Construction via Reinforcement Learning**
  - 用途：把「記憶何時介入/怎麼寫入」從 heuristic 變成可學習 policy（可映射到 L2 memory gating）。
  - 歸檔筆記：`docs/literature/MEM-ALPHA-2509.25911.md`
  - 優先級：⭐⭐⭐⭐

### B4. Memory management + usage via RL（Memory-R1）
- `4c8c4f49-118e-4740-a142-3302245834ec.pdf`
  - Title: **Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning**
  - 用途：把 memory manager（ADD/UPDATE/DELETE/NOOP）與 answer-time memory distillation 都納入 RL，提供「可審計記憶操作 + outcome-driven reward」設計模板。
  - 歸檔筆記：`docs/literature/MEMORY-R1-2508.19828.md`
  - 優先級：⭐⭐⭐⭐

### B5. RL backbone 選型基準（task ↔ architecture 對照）
- `1d22097a-9f22-4bde-af51-fadbf6df84fe.pdf`
  - Title: **RLBenchNet: The Right Network for the Right Reinforcement Learning Task**
  - 用途：提供「任務型態→模型架構」選型規則與計算成本對照（SPS / latency / GPU memory），可直接用來定義 L2 planner 的分級實驗路線。
  - 歸檔筆記：`docs/literature/RLBENCHNET-2505.15040.md`
  - 優先級：⭐⭐⭐⭐

## C. L3（低層 follower / 穩定執行）與機器人實驗證據

### C1. Robotics empirical/control paper（待精讀）
- `d2fa8f71-741b-49f4-b3f3-4f0c3080ccf4.pdf`
  - First-page source: *Frontiers in Robotics and AI (2025)*
  - 用途：補低層執行/robustness 實驗設計與評估指標。
  - 優先級：⭐⭐⭐

## D. 安全與約束學習（橫跨 L2/L3）

### D1. Lyapunov-stable actor-critic（LAC）
- `78f52902-f3d2-4a81-a0e1-0eec28e7fb26.pdf`
  - Title: **Actor-Critic Reinforcement Learning for Control with Stability Guarantee**
  - 用途：提供可 sample 驗證的 Lyapunov stability 條件與 actor-critic 約束訓練框架，可直接映射到 L2 的 safety-critic / constraint loss 設計。
  - 歸檔筆記：`docs/literature/LAC-STABILITY-ACTOR-CRITIC-2020.md`
  - 優先級：⭐⭐⭐⭐⭐

### D2. 待補主線
- CPO / CMDP 經典論文（尚待補齊 PDF 與實作對照）
- 已在記憶中列為主線：`memory/2026-02-17.md`

## E. 非研究論文（不列入架構證據）

以下文件保留但不作為研究證據源：
- 履歷/cover letter 類：`1a5d...`, `2e66...`, `a729...`, `d501...`, `d503...`
- 作業說明/課內文件：`ba1e...`
- OpenClaw 協議與個人規劃：`1b6e...`, `3999...`, `5662...`, `ce57...`

## 使用方式（入口）
1) 先看 `docs/literature/PIPELINE_GUIDE.md`（按環節找文獻）
2) 針對當前改版只選 1~2 篇主文獻做實作對齊
3) 每次實驗報告要標註「此改動對齊哪篇文獻」
