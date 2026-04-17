# V5.1 Structure Convention

## 1) 命名空間與主實作位置
- V5.1 主實作統一放在：
  - `hrl_ws/src/hrl_trainer/hrl_trainer/v5_1/`
- 新增功能、重構、實驗分支皆以 `v5_1` 為目錄根，不再新增 `task1` 命名。

## 2) 舊路徑策略（相容但不擴充）
- 舊路徑 `v5/task1_*` 僅維持 backward compatibility：
  - 允許 bugfix（blocking/安全問題）
  - 不允許新增功能、API 擴展、實驗邏輯
- 若需橋接，請用 adapter/shim 指向 `v5_1`，並在 PR 註明 deprecation note。

## 3) 推薦 artifacts 路徑（V5.1）
- 統一根目錄：`artifacts/v5_1/`
- 建議子路徑：
  - `artifacts/v5_1/contracts/`
  - `artifacts/v5_1/l2_sac/`
  - `artifacts/v5_1/l3_executor/`
  - `artifacts/v5_1/safety_watchdog/`
  - `artifacts/v5_1/curriculum/`
  - `artifacts/v5_1/e2e/<run_id>/`
  - `artifacts/v5_1/logs/l1|l2|l3/`
  - `artifacts/v5_1/gates/`

## 4) 目錄治理規則（最小）
- 所有 V5.1 pipeline 產出必須能落到 `artifacts/v5_1/...`。
- 每次 run 必須有 `run_id`，且能對應 L1/L2/L3 分層 log。
- 文件與腳本引用新路徑時，優先使用 `v5_1`；舊 `task1` 只作讀取相容。

## 5) 遷移原則
- 先 freeze 契約，再搬遷實作，再移除冗餘相依。
- 不做一次性大搬家；採可回退的小步遷移。
- 每次遷移需附 rollback 說明與驗證命令。
