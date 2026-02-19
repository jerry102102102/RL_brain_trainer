# V3 Guide (實驗與程式總覽)

這份文件是 V3 主線的入口，給「要快速接手」的人看。

## 1) 核心架構
- L1：語意/全域理解（輸出意圖與約束，不直接輸出控制）
- L2：局部規劃/技能（可含 memory/LSTM）
- L3：deterministic follower（穩定追蹤）

## 2) 主要腳本
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/train_rl_brainer_v3_online.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/run_v3_hierarchy_meaning_ablation.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/diag_l0_rbf_straight.py`
- `hrl_ws/src/hrl_trainer/hrl_trainer/sim2d/diag_l1_planner_straight.py`

## 3) 主要設定檔
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_online_quick.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_hierarchy_meaning_ablation.yaml`
- `hrl_ws/src/hrl_trainer/config/train_rl_brainer_v3_level5_pentagon_ablation.yaml`

## 4) 重要報告
- `V3_HIERARCHY_MEANING_ABLATION.md`
- `V3_LEVEL5_PENTAGON_ABLATION.md`
- `V3_COMPLEXITY_ABLATION.md`
- `V3_THREE_LAYER_LSTM_ABLATION.md`
- `L2_MEMORY_ABLATION.md`
- `L2_DETERMINISTIC_PLUS_MEMORY.md`

## 5) 建議跑法（最短路）
1. 跑 L0/L1 診斷，先確認底座
2. 跑 hierarchy meaning ablation
3. 跑 level5 pentagon
4. 把結果整理回 docs + artifact（json）

## 6) 目前結論（摘要）
- L2 在中高難度（Level1~4）有效
- Level5 接近飽和區，需要優先補「可行性表徵 + 約束規劃骨架」
- memory/LSTM 不應覆蓋 deterministic core
