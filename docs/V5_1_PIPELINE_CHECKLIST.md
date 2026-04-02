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

### D.1 當前已驗證可用的場景啟動方式（Jerry 實測）

> Jerry 的 shell 是 `zsh`。若 agent 端預設是 `bash`，不得直接照搬錯誤 `.bash` / `src/install` 路徑；需以目前已驗證可用的 source 流程為準。

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
scripts/v5/launch_kitchen_scene.sh --mode gui
```

如果要 headless：

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
scripts/v5/launch_kitchen_scene.sh --mode headless
```

### D.2 當前已驗證可用的 GZ pipeline 執行命令（Jerry 實測）

```bash
cd /home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
source hrl_ws/.venv/bin/activate
export PYTHONPATH=/home/jerry/.openclaw/workspace/repos/personal/RL_brain_trainer/hrl_ws/src/hrl_trainer:$PYTHONPATH

python -m hrl_trainer.v5_1.pipeline_e2e \
 --run-id main_reward_gz_30step_smoke \
 --episodes 1 \
 --steps-per-episode 30 \
 --artifact-root artifacts/v5_1/e2e/main_reward_gz_30step_smoke \
 --runtime-mode gz \
 --policy-mode sac_torch \
 --stage-profile s0_b \
 --target-mode near_home \
 --runtime-joint-names Rack_joint,robot_base_joint,shoulder1_joint,shoulder2_joint,wr1_joint,wr2_joint,wr3_joint \
 --trajectory-topic /arm_controller/joint_trajectory \
 --joint-state-topic /joint_states
```

### D.3 每次跑完後的固定解讀順序

必讀 artifact：
- `artifacts/v5_1/e2e/<run_id>/gate_result.json`
- `artifacts/v5_1/e2e/<run_id>/pipeline_summary.json`
- `artifacts/v5_1/e2e/<run_id>/episode_reward_summary.jsonl`
- 視需要加看：`reward_trace.jsonl`、`runtime_trace.jsonl`

解讀順序：
1. **先看這次是不是「真的跑起來」**
   - `episodes_completed == episodes_requested`
   - `reset_failures == 0`
   - `execution_ratio == 1.0`
   - `l1/l2/l3` logs 有值
2. **再看 gate 有沒有過**
   - `gate_overall_decision`
   - `gate_passed`
3. **最後才看任務/訓練表現**
   - `done_reason`（`success` / `timeout` / `execution_fail`）
   - `success_rate`
   - `component_sums.progress`
   - `near_goal`
   - `dwell`
   - `success_bonus`
   - `reward_total`

### D.4 一個重要規則
- `exit_code = 0` 只代表命令執行完，不代表實驗成功。
- 若 `gate_passed = false` 或 `overall_decision = HOLD`，回報時必須明確標記為 **FAIL/HOLD**，不可說成「成功」。

### D.5 執行完訓練後第一件事：清理 GZ 相關進程

```bash
pkill -f "ros2 launch kitchen_robot_description gazebo.launch.py|gz sim|parameter_bridge|tray_pose_adapter|controller_manager" || true
sleep 2
```

這一步是固定規則，避免殘留 process 汙染下一輪實驗。

## E) 驗收判定（最低）
- Bring-up/health check 全 PASS。
- E2E pipeline 命令 exit code = 0。
- `L1/L2/L3` 三層 log 均存在且欄位完整。
- 若任一條件未滿足：本次 run 標記 FAIL，不得宣稱可重跑通過。
