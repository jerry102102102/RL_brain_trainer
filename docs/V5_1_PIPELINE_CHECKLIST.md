# V5.1 Pipeline Bring-up & Layer Logging Checklist

> 原則：每次 pipeline 都要先 bring-up/health check；每層輸出必記錄（L1/L2/L3）。

## A) Pipeline 前置（每次必跑）

### A.1 環境 bring-up
1. 載入 ROS 與 workspace 環境。
2. 啟動場景/控制器（headless 或 gui）。
3. 等待控制器 ACTIVE。

### A.2 最小健康檢查
1. `controller_manager` 中關鍵 controller 為 `active`。
2. 感知鏈 topic 可讀（至少一次 sample）：`/tray1/pose`、`/v5/perception/object_pose_est`。
3. L1/L2/L3 topic 存活（topic list/echo 一次）。
4. 產生 `run_id` 並建立 `artifacts/v5_1/e2e/<run_id>/`。

> 任一項 fail：不得執行正式 pipeline，先修復環境。

## B) L1/L2/L3 每層最小必記欄位

### L1（Intent）
- `run_id`
- `timestamp`
- `intent_id`
- `task_context`
- `constraints`
- `output_intent_packet`
- `status` (ok/fail)

### L2（Policy / Skill）
- `run_id`
- `timestamp`
- `policy_version`
- `obs_digest`（可摘要）
- `skill_command`
- `confidence_or_value`
- `latency_ms`
- `status` (ok/fail)

### L3（Executor / Safety）
- `run_id`
- `timestamp`
- `executor_version`
- `input_skill_command`
- `clamp_or_guard_action`
- `final_controller_cmd`
- `watchdog_state`
- `latency_ms`
- `status` (ok/fail)

## C) Fail 定位流程（先看哪層）
1. **先看 L3**：是否有 safety clamp/watchdog timeout/controller reject。
2. **再看 L2**：是否 command 無效、頻率不符、延遲過大。
3. **最後看 L1**：intent/context 是否錯誤導致下游全錯。
4. 同步對照 `run_id` 下三層 log，禁止只看單層就定因。

## D) 最低驗收命令模板

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source scripts/v5_1/activate_env.sh
scripts/v5_1/env_check.sh

# 1) bring-up + 基礎健康檢查
scripts/v5/launch_kitchen_scene.sh --mode headless
ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers "{}"
ros2 topic echo /tray1/pose --qos-reliability best_effort --once
ros2 topic echo /v5/perception/object_pose_est --qos-reliability best_effort --once

# 2) V5.1 pipeline（範本）
bash scripts/v5_1/run_e2e_pipeline.sh --bringup --verify --run-id <run_id>

# 3) 分層 log 驗證（範本）
python scripts/v5_1/validate_layer_logs.py --run-id <run_id>
```

## E) 驗收判定（最低）
- Bring-up/health check 全 PASS。
- E2E pipeline 命令 exit code = 0。
- `L1/L2/L3` 三層 log 均存在且欄位完整。
- 若任一條件未滿足：本次 run 標記 FAIL，不得宣稱可重跑通過。
