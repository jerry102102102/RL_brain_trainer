# HRL Workspace Guide (V3-Only)

這個 `hrl_ws` 現在是 **V3 研究主線專用工作區**，已移除舊的 Gazebo/bridge 路徑，專注在 `sim2d` 的分層實驗。

## 目前保留內容

```text
hrl_ws/
├── pyproject.toml
└── src/hrl_trainer/
    ├── config/                      # V3 訓練與消融設定檔
    ├── hrl_trainer/sim2d/           # V3 核心訓練/診斷/ablation 腳本
    └── ...
```

## 移除內容（legacy）
- `src/hrl_gazebo/`
- `src/hrl_control_bridge/`

> 目的：避免舊 ROS/Gazebo 路徑干擾目前 V3 研究與 final-project 輸出。

---

## 快速開始

1) 進入工作區
```bash
cd hrl_ws
```

2) 建立環境
```bash
uv sync
source .venv/bin/activate
```

3) 進行 V3 實驗（依 config 跑）
- 先看：`src/hrl_trainer/config/`
- 核心腳本：`src/hrl_trainer/hrl_trainer/sim2d/`

---

## 建議執行順序（V3）
1. L0/L1 診斷（先確認層級可用）
2. hierarchy meaning ablation
3. level5 pentagon 壓力測試

對應報告請更新到 repo root 的 `docs/`。

---

## 你應該先看哪裡
- Repo 主說明：`../README.md`
- V3 實驗總覽：`../docs/V3_GUIDE.md`
- 進度板：`../docs/V3_PROGRESS_BOARD.md`
