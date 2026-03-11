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

執行 V4（範例）
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

### 目前建議流程
1. V5：先確認 `../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` 的當前 WP 狀態（WP0/WP1/WP1.5/WP2）
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

Run V4 (example)
```bash
python src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py \
  --config src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml
```

### Recommended Current Workflow
1. V5: check current WP status in `../docs/V5_KITCHEN_IMPLEMENTATION_PLAN.md` first.
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
