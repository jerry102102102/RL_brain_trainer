# HRL Workspace Guide (V5 Active / V4 Legacy)

## 中文版

`hrl_ws` 是本專案的實驗工作區，當前主軸為 **V5 manipulation**，同時保留 **sim2d V4** 作為 legacy baseline。

### Workspace 結構
```text
hrl_ws/
├── pyproject.toml
└── src/hrl_trainer/
    ├── config/                    # v4 + v5 設定（WP0/WP1/WP1.5/WP2 等）
    ├── hrl_trainer/sim2d/         # V4 sim2d 環境、訓練、評估與可視化
    └── hrl_trainer/v5/            # V5 三層架構模組（intent/skill/control 與相關工具）
```

### 快速開始
```bash
cd hrl_ws
uv sync
source .venv/bin/activate
```

### 當前維護中的 V5.1 / 5.2 主線
- 目前這條分支的實際維護主線是：
  - `src/hrl_trainer/hrl_trainer/v5_1/pipeline_e2e.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/sac_torch.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/reward.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/build_teacher_dataset.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/train_deterministic_student.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/eval_deterministic_student.py`
- 這條主線同時包含：
  - V5.1：SAC teacher + deterministic safety execution
  - 5.2：teacher-student deterministic extraction
- 階段性總結文件在 repo root：
  - `../V5_1_STAGE_SUMMARY.md`
  - `../V5_1_EXECUTIVE_SUMMARY.md`
  - `../V5_1_TIMELINE.md`

執行 V4（範例）
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

執行 V5 M2.3（Reward Composer v2 + Curriculum A/B/C）最小測試
```bash
cd hrl_ws
source .venv/bin/activate
PYTHONPATH=src/hrl_trainer python -m unittest src/hrl_trainer/tests/test_v5_m2_3_reward_curriculum.py -v
```

執行 V5 M2.3a integration smoke（少量 step，顯示 stage 與 reward component）
```bash
cd hrl_ws
source .venv/bin/activate
PYTHONPATH=src/hrl_trainer python -m hrl_trainer.v5.trainer_loop --episode-index 1200 --steps 3 --terminal-success
```

預期輸出欄位（範例）
```text
stage_id=B
reward_term_totals={'sparse_terminal': 2.0, 'pbrs_delta': ..., 'safety_penalty': ..., 'smoothness_penalty': ..., 'coverage': ..., 'subgoal': ...}
step_0_weighted_terms={...}
```

### WP2（M2-7~M2-9）現況快照
- WP2 closeout 已完成於 **benchmark/eval harness real path**，含：
  - M2-7 training loop integration
  - M2-8 baseline benchmark
  - M2-8a eval harness strict-policy path
  - M2-8b 4-variant formal comparison
  - M2-9 one-click rerun script
- ⚠️ 以上 **不是 real robot runtime / HIL**；目前未宣稱實機驗證完成。
- 參考：`../docs/WP2_IMPLEMENTATION_NOTE.md`
- 一鍵重跑：`../scripts/m2_9_rerun_wp2.sh`

### 目前建議流程
1. V5：先確認 `../docs/WP2_IMPLEMENTATION_NOTE.md` 與 `../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` 的當前狀態
2. V4：若需 baseline 對照，再跑 no-obstacle/complex/velocity constraints 三組實驗
3. 所有新實驗請記錄到 `../docs/V5_EXPERIMENT_LOG.md`

主要 config：
- `src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### 控制約束（現行）
- `v ∈ [-1.2, 1.2]`
- `ω ∈ [-2π, 2π]`（±360 deg/s）

### 延伸文件
- Repo 主說明：`../README.md`
- V5 主計畫：`../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md`
- V5 設計哲學：`../docs/V5_DESIGN_PHILOSOPHY.md`
- V4 介面規範：`../docs/V4_INTERFACE_SPEC.md`
- 文獻索引：`../docs/literature/INDEX.md`
- V3 歷史資料：`../docs/V3_GUIDE.md`

---

## English Version

`hrl_ws` is the experiment workspace for this project, now **V5-active (manipulation)** with **sim2d V4** kept as a legacy baseline.

### Workspace Layout
```text
hrl_ws/
├── pyproject.toml
└── src/hrl_trainer/
    ├── config/                    # v4 + v5 configs (WP0/WP1/WP1.5/WP2, etc.)
    ├── hrl_trainer/sim2d/         # V4 sim2d env, training, evaluation, visualization
    └── hrl_trainer/v5/            # V5 three-layer modules (intent/skill/control + tools)
```

### Quick Start
```bash
cd hrl_ws
uv sync
source .venv/bin/activate
```

### Current maintained V5.1 / 5.2 line
- The actively maintained path on this branch is:
  - `src/hrl_trainer/hrl_trainer/v5_1/pipeline_e2e.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/sac_torch.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/reward.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/build_teacher_dataset.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/train_deterministic_student.py`
  - `src/hrl_trainer/hrl_trainer/v5_1/eval_deterministic_student.py`
- This maintained line now covers both:
  - V5.1: SAC teacher + deterministic safety execution
  - 5.2: teacher-student deterministic extraction
- Stage-close summaries live at the repo root:
  - `../V5_1_STAGE_SUMMARY.md`
  - `../V5_1_EXECUTIVE_SUMMARY.md`
  - `../V5_1_TIMELINE.md`

Run V4 (example)
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

Run V5 M2.3 minimal tests (Reward Composer v2 + Curriculum A/B/C)
```bash
cd hrl_ws
source .venv/bin/activate
PYTHONPATH=src/hrl_trainer python -m unittest src/hrl_trainer/tests/test_v5_m2_3_reward_curriculum.py -v
```

Run V5 M2.3a integration smoke (few steps; prints stage + reward components)
```bash
cd hrl_ws
source .venv/bin/activate
PYTHONPATH=src/hrl_trainer python -m hrl_trainer.v5.trainer_loop --episode-index 1200 --steps 3 --terminal-success
```

Expected output fields (example)
```text
stage_id=B
reward_term_totals={'sparse_terminal': 2.0, 'pbrs_delta': ..., 'safety_penalty': ..., 'smoothness_penalty': ..., 'coverage': ..., 'subgoal': ...}
step_0_weighted_terms={...}
```

### WP2 (M2-7~M2-9) status snapshot
- WP2 closeout is complete on the **benchmark/eval harness real path**, including:
  - M2-7 training loop integration
  - M2-8 baseline benchmark
  - M2-8a strict-policy eval harness path
  - M2-8b 4-variant formal comparison
  - M2-9 one-click rerun script
- ⚠️ This is **not** real robot runtime / HIL validation.
- Reference: `../docs/WP2_IMPLEMENTATION_NOTE.md`
- One-click rerun: `../scripts/m2_9_rerun_wp2.sh`

### Recommended Current Workflow
1. V5: check `../docs/WP2_IMPLEMENTATION_NOTE.md` and `../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` first.
2. V4: run no-obstacle/complex/velocity suites only when a baseline comparison is needed.
3. Log all new experiments in `../docs/V5_EXPERIMENT_LOG.md`.

Main configs:
- `src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### Current Control Constraints
- `v ∈ [-1.2, 1.2]`
- `ω ∈ [-2π, 2π]` (±360 deg/s)

### Related Docs
- Main repo README: `../README.md`
- V5 implementation plan: `../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md`
- V5 design philosophy: `../docs/V5_DESIGN_PHILOSOPHY.md`
- V4 interface spec: `../docs/V4_INTERFACE_SPEC.md`
- Literature index: `../docs/literature/INDEX.md`
- V3 historical docs: `../docs/V3_GUIDE.md`
