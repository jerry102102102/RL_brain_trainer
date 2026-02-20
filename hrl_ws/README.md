# HRL Workspace Guide (V4)

## 中文版

`hrl_ws` 是本專案的實驗工作區，現行主線為 **sim2d V4**。

### Workspace 結構
```text
hrl_ws/
├── pyproject.toml
└── src/hrl_trainer/
    ├── config/                    # v2/v3/v4 訓練設定（目前主跑 v4）
    └── hrl_trainer/sim2d/         # sim2d 環境、訓練、評估與可視化
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

### V4 建議流程
1. 先跑 no-obstacle MVP（確認基本收斂）
2. 再跑 complex MVP（確認多障礙穩定性）
3. 最後跑 complex velocity constraints（驗證控制邊界）

主要 config：
- `src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### 控制約束（現行）
- `v ∈ [-1.2, 1.2]`
- `ω ∈ [-2π, 2π]`（±360 deg/s）

### 延伸文件
- Repo 主說明：`../README.md`
- V4 介面規範：`../docs/V4_INTERFACE_SPEC.md`
- 文獻索引：`../docs/literature/INDEX.md`
- V3 歷史資料：`../docs/V3_GUIDE.md`

---

## English Version

`hrl_ws` is the experiment workspace for this project, with **sim2d V4** as the current mainline.

### Workspace Layout
```text
hrl_ws/
├── pyproject.toml
└── src/hrl_trainer/
    ├── config/                    # v2/v3/v4 training configs (focus on v4)
    └── hrl_trainer/sim2d/         # sim2d env, training, evaluation, visualization
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

### Recommended V4 Workflow
1. Run no-obstacle MVP first (basic convergence check)
2. Run complex MVP (multi-obstacle robustness)
3. Run complex velocity constraints (control-bound validation)

Main configs:
- `src/hrl_trainer/config/train_rl_brainer_v4_no_obstacle_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp.yaml`
- `src/hrl_trainer/config/train_rl_brainer_v4_complex_mvp_velocity.yaml`

### Current Control Constraints
- `v ∈ [-1.2, 1.2]`
- `ω ∈ [-2π, 2π]` (±360 deg/s)

### Related Docs
- Main repo README: `../README.md`
- V4 interface spec: `../docs/V4_INTERFACE_SPEC.md`
- Literature index: `../docs/literature/INDEX.md`
- V3 historical docs: `../docs/V3_GUIDE.md`
