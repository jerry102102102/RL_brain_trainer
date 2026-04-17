# WP0 Script Canonical Entry (2026-03)

## 唯一推薦入口（scene launch）
- **Primary (推薦)：** `scripts/v5/launch_kitchen_scene.sh`
- 原因：與 `scripts/v5/bridge_kitchen_scene.sh` 的 `external/kitchen_scene` 橋接路徑一致，且支援 `--mode headless|gui`。

## 備援入口（僅在主入口不可用時）
- **Fallback：** `external/ENPM662_Group4_FinalProject/scripts/wp0_smoke.sh`
- 用途：在 scene repo 內做最小 smoke（build + launch + trajectory trigger）。
- 注意：這是 scene repo 內部腳本，不是 RL repo 的首選入口。

## GUI / headless 差異
- **Headless（CI/遠端推薦）**
  - `scripts/v5/launch_kitchen_scene.sh --mode headless`
  - 預設包含 `headless:=true use_software_renderer:=true`
- **GUI（本機桌面除錯）**
  - `scripts/v5/launch_kitchen_scene.sh --mode gui`
  - 會使用 `headless:=false`

## Smoke / Recheck 建議
- RL repo smoke：`scripts/wp0_scene_smoke.sh`
  - tray topic 兼容：`/tray_tracking/pose_stream_raw` 或 `/tray_tracking/pose_stream`
- scene repo recheck：`external/ENPM662_Group4_FinalProject/scripts/wp0_recheck.sh`
  - 若無 `rg` 會自動 fallback `grep`，不再 hard fail。

## 不再建議
- 直接以 `external/ENPM662_Group4_FinalProject` 作為 RL 主入口路徑。
- 在 README/Runbook 直接貼 `ros2 launch ...` 當作第一入口（容易與橋接路徑分岔）。
