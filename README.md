# Hierarchical RL Controller — README (中英雙語 / Bilingual)

> Single-file module: **`hrl_control.py`** → `from hrl_control import HierarchicalRLController, HRLConfig`

* Repo focus: **三層式分工：概念層(離散RL) → 技能層(DMP) → 安全層(CBF-QP)**；支援 **CPU/GPU**、**ROS 2（選用）**、**可重現訓練**、**易於擴充**。
* This README is bilingual. For each section, **中文在前 / English follows**.

---

## 目錄 / Table of Contents

> Project roadmap note: v0 final-project architecture plan is tracked in `docs/V0_ARCHITECTURE.md`.
> Research-mode upgrade: three-layer v2 design spec is tracked in `docs/THREE_LAYER_ARCH_V2.md`.

1. [模組一：整體架構說明 / Module 1: Overall Architecture](#模組一整體架構說明--module-1-overall-architecture)

   * 發想、數學、架構、實現、優勢 / Motivation, Math, Design, Implementation, Advantages
2. [模組二：數學可行性與論文來源 / Module 2: Mathematical Feasibility & References](#模組二數學可行性與論文來源--module-2-mathematical-feasibility--references)
3. [模組三：程式碼與使用指南 / Module 3: Code Guide & Usage](#模組三程式碼與使用指南--module-3-code-guide--usage)

---

## 模組一：整體架構說明 / Module 1: Overall Architecture

### 中文

**發想**

* 受生物系統啟發：高層只處理**低維概念/目標**（像大腦），具體運動由**技能庫**（類似小腦/脊髓）生成，最後交給**安全層**在物理限制下做最小修正。
* 問題被分解為：**概念決策**（離散 options）→ **連續運動合成**（DMP）→ **安全約束投影**（CBF-QP）。

**數學**

* **概念層（DQN/Double DQN）**：在抽象狀態上選擇**離散選項**（macro-actions）。
* **技能層（DMP）**：對每關節用 RBF 基底逼近的強迫項產生平滑軌跡；時間尺度嚴格使用 $\tau^2\ddot{x}$ 與 $v=\tau\dot{x}$。
* **安全層（CBF-QP）**：以二次規劃最小化 $\|u-u_{des}\|_R^2$，並滿足 joint 速度/位置界與 **CBF**：
  $\nabla h(x)^\top (f(x)+g(x)u) + \alpha h(x) \ge 0.$

**架構**

* `HierarchicalRLController` 組合：

  * `QNetwork`（policy/target）、replay、epsilon 調度、**Double DQN**（可切換）
  * `DMPModel`（基底均勻於相位 `s`、可擬合/展開）
  * CBF-QP 安全層（OSQP 首選，`qpsolvers` 後備；**bounds/CBF 皆可加 slack**）
  * `OptionSpec`：結構化定義 macro-action（目標偏移比例、時間縮放、技能族…）
  * 選配 ROS 2：`run_ros2_closed_loop()`（`~/set_goal` 設定目標）

**實現**

* 單檔模組便於移植；**lazy import** 讓沒裝 ROS 也能 `import`。
* `goal_in_state=True` 時，自動把 **誤差 `g-q` 與 `g`** 拼進觀測，並支援 **StateNormalizer** 跑動態正規化。
* QP 不可行時提供**回退策略**（縮 `u_des` / 調 slack 權重 / 最終裁剪）。

**優勢**

* **學習穩定**：macro-action 縮短決策地平線，Double DQN 抑制高估，輸入正規化穩態。
* **可證安全**：CBF 保障安全集合的前向不變性（在可行且無 slack 情況）。
* **工程可用**：單檔、ROS 2 介面、完整 save/load、可插拔 barrier。

---

### English

**Motivation**

* Bio-inspired separation: high-level **concepts/goals** vs. low-level **motor skills**, with a **safety layer** enforcing physical limits.

**Math**

* **Concept (DQN/Double DQN)** chooses **discrete options** over abstract states.
* **Skill (DMP)** synthesizes smooth joint trajectories using RBF-based forcing terms with correct time scaling $\tau^2\ddot{x}$, $v=\tau\dot{x}$.
* **Safety (CBF-QP)** minimally modifies $u_{des}$ subject to joint bounds and the CBF inequality $\nabla h^\top(f+gu)+\alpha h\ge 0$.

**Design**

* `HierarchicalRLController` composes policy/target nets, replay, epsilon scheduling, **Double DQN**, `DMPModel`, and a **CBF-QP** filter (OSQP primary, `qpsolvers` fallback; **slack for bounds/CBF**).
* `OptionSpec` parameterizes macro-actions; optional ROS 2 loop with `~/set_goal`.

**Implementation**

* Single-file, lazy ROS imports; `goal_in_state=True` augments observation with **`g-q` and `g`**, with a **running normalizer**.
* QP infeasibility triggers a **backoff strategy** (shrink `u_des`, relax bound slack, final clipping).

**Advantages**

* **Stability**: options shorten horizons; Double DQN mitigates overestimation; normalized inputs.
* **Safety**: CBF ensures forward invariance when constraints are feasible without slack.
* **Practicality**: single file, ROS 2 integration, full save/load, pluggable barriers.

---

## 模組二：數學可行性與論文來源 / Module 2: Mathematical Feasibility & References

### 中文

**DMP（Dynamic Movement Primitives）**

* 相位：$\tau\dot{s}=-\alpha_s s$。
* 變換系統：$\tau\dot{v}=\alpha_z(\beta_z(g-x)-v)+(g-x_0)f(s),\;\tau\dot{x}=v$。
* 擬合：以 RBF 基底 $\psi_i(s)=\exp(-h_i(s-c_i)^2)$ 做 LWR，
  $f(s)=\frac{\sum_i \psi_i(s) w_i s}{\sum_i \psi_i(s)+\varepsilon}.$
* 從示範恢復強迫項：
  $v_{demo}=\tau\dot{x},\quad f^*=\frac{\tau^2\ddot{x}-\alpha_z(\beta_z(g-x)-v_{demo})}{(g-x_0)+\varepsilon}.$
* **性質**：當 $f\equiv0$ 時是臨界阻尼收斂到 $g$；RBF 只調整形狀，穩定性主要由線性部份決定。

**CBF-QP（Control Barrier Function）**

* 安全集合 $\mathcal{S}=\{x\mid h(x)\ge 0\}$ 的前向不變性條件：
  $\nabla h(x)^\top(f(x)+g(x)u)+\alpha h(x)\ge 0.$
* 我們把 `u_des` 投影到滿足線性不等式與邊界的集合內；問題是**凸 QP**，OSQP 能高效解。
* 若不可行，引入 slack（軟化）確保穩定求解；slack 的大小量化了約束“違反”的必要程度。

**DQN 與 Double DQN**

* Bellman 目標：$y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')$。
* **Double DQN** 將 argmax 與估值分離：
  $a^*=\arg\max_a Q_\theta(s',a),\; y=r+\gamma Q_{\theta^-}(s',a^*)$。
* 在函數逼近與 off-policy 設定下沒有強收斂保證，但實務上能顯著降高估與不穩定。

**階層式 RL（Options/SMDP）**

* 使用宏動作（options）將長視野分解為若干短期子任務，有助於探索與樣本效率；與 DMP 搭配可把連續控制“封裝”成可重用技能。

**參考文獻**（建議起點）

* Ijspeert, Nakanishi, Schaal. *Movement Imitation with DMPs* / *T-NNLS 2013*；Schaal et al., 2006（早期）
* Ames et al. *Control Barrier Functions: Theory and Applications*, 2016（Tutorial/Survey）
* Mnih et al. *Human-level control through deep RL*, *Nature 2015*（DQN）
* Van Hasselt et al. *Deep Reinforcement Learning with Double Q-learning*, *AAAI 2016*
* Sutton, Precup, Singh. *Between MDPs and SMDPs: A framework for temporal abstraction in RL*, *AIJ 1999*（Options）
* Kober, Bagnell, Peters. *RL in Robotics: A Survey*, *IJRR 2013*

> 註：本模組不內建運動學/碰撞距離的解析式；若需幾何 CBF，請提供對應的 $h(x)$ 與梯度近似/解析式。

---

### English

**DMP**

* Canonical: $\tau\dot{s}=-\alpha_s s$.
* Transformation: $\tau\dot{v}=\alpha_z(\beta_z(g-x)-v)+(g-x_0)f(s),\; \tau\dot{x}=v$.
* RBF forcing: $f(s)=\frac{\sum_i\psi_i(s) w_i s}{\sum_i\psi_i(s)+\varepsilon}$, $\psi_i=\exp(-h_i(s-c_i)^2)$.
* From demos: $v_{demo}=\tau\dot{x}$, $f^*=\frac{\tau^2\ddot{x}-\alpha_z(\beta_z(g-x)-v_{demo})}{(g-x_0)+\varepsilon}$.

**CBF-QP**

* Forward invariance for $\mathcal{S}=\{x\mid h(x)\ge0\}$: $\nabla h^\top(f+gu)+\alpha h\ge 0$.
* We solve a convex QP projecting $u_{des}$ onto constraints; slack ensures feasibility and quantifies minimal violation.

**DQN / Double DQN**

* Bellman target: $y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')$; Double DQN uses online argmax + target evaluation.

**Hierarchy / Options**

* Macro-actions shorten horizons and improve exploration; pairing with DMPs yields reusable continuous skills.

**References**

* Ijspeert et al., 2013; Schaal et al., 2006 (DMPs)
* Ames et al., 2016 (CBF)
* Mnih et al., 2015 (DQN)
* Van Hasselt et al., 2016 (Double DQN)
* Sutton, Precup, Singh, 1999 (Options)
* Kober et al., 2013 (RL in Robotics survey)

---

## 模組三：程式碼與使用指南 / Module 3: Code Guide & Usage

### 安裝 / Installation

```bash
pip install torch numpy scipy osqp qpsolvers
# ROS 2 packages are optional; install per your distro if needed.
```

### 匯入與核心類別 / Import & Core Class

```python
from hrl_control import HierarchicalRLController, HRLConfig, OptionSpec
```

* 單檔 `hrl_control.py` 提供：

  * `HierarchicalRLController`：高層策略 + DMP + CBF-QP
  * `HRLConfig`：超參數
  * `OptionSpec`：macro-action 規格
  * `ToyJointEnv`：DMP 驅動的 smoke test 環境

### 狀態維度計算 / State Dimension Formula

若 `goal_in_state=True`（預設）且基礎觀測 `base_state` 維度為 `B`，關節數 `n_joints=J`，
實際網路輸入維度：

$\text{state\_dim} = B + 2J\quad (\text{concatenate } [\,base\_state,\; g-q,\; g\,]).$

### 快速開始（Toy 環境）/ Quick Start (Toy Env)

```python
import numpy as np
from hrl_control import HierarchicalRLController, HRLConfig, ToyJointEnv

env = ToyJointEnv()
state_dim = env.state_dim + 2 * env.n_joints   # if goal_in_state=True
cfg = HRLConfig(
    n_joints=env.n_joints,
    state_dim=state_dim,
    options=["hold", "dmp_small_step", "dmp_medium_step", "dmp_large_step"],
    horizon_steps=6,
    dt=0.1,
    batch_size=32,
    replay_capacity=2000,
    eps_decay_steps=1000,
    seed=0,
)
ctrl = HierarchicalRLController(cfg)

# 擬合一段示範給 DMP（每列是一個時間步，欄=關節）
demo = np.linspace(0.0, float(env.goal[0]), 50, dtype=np.float64)[:, None]
ctrl.dmp_fit_weights(demo, T=cfg.dt * (demo.shape[0] - 1))

# 訓練與評估
train_metrics = ctrl.train(env, total_steps=500, warmup=50, log_interval=200)
print("train:", train_metrics)
print("eval:", ctrl.evaluate(env, episodes=2))
```

### 介接你自己的模擬（Gazebo / ROS 以外）/ Wrap Your Own Sim

實作 `EnvProtocol`（不需 gym）：

```python
import numpy as np
from typing import Tuple

class MyGazeboEnv:  # implements EnvProtocol
    def __init__(self):
        self._n_joints = 6
        self._goal = np.zeros(self._n_joints, dtype=np.float32)
        # ... init sim ...

    def reset(self) -> np.ndarray:
        # return base_state (e.g., [q, dq])
        return np.zeros(2*self._n_joints, dtype=np.float32)

    def step(self, option_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Execute macro-action in your sim (option→trajectory→apply)
        # Return (next_state, reward, done, {"goal": current_goal_np})
        return np.zeros(2*self._n_joints, dtype=np.float32), 0.0, False, {"goal": self._goal}

    @property
    def state_dim(self) -> int:
        return 2*self._n_joints

    @property
    def n_joints(self) -> int:
        return self._n_joints

    @property
    def goal(self) -> np.ndarray:
        return self._goal
```

* 計算 `state_dim = env.state_dim + 2 * env.n_joints`（若 `goal_in_state=True`）。
* 在 `step()` 裡，建議**真的執行**所選 option 對應的運動（例如讓模擬器走一小段 DMP），回傳新的狀態與 reward。

### 設定 DMP 與示範 / DMP Fitting

* 示範張量形狀：`[T_steps, n_joints]`，時間長度 `T = dt * (T_steps - 1)`。
* 你可以為多關節一次擬合；每關節各自有權重，但共享相位 `s` 與基底中心/寬度。

### 安全層與 Barrier / Safety Layer & Barriers

* 內建：**關節上下界**轉為 CBF；同時也有**速度/一步位置界**。
* 自訂 barrier：

```python
import numpy as np

def sphere_barrier(q: np.ndarray) -> tuple[float, np.ndarray]:
    # 假設 1-DoF 例子：離某危險位置 q_obs 的距離 d(q)=|q - q_obs|
    q_obs = 1.2
    min_dist = 0.3
    h = (np.abs(q[0] - q_obs) - min_dist)  # h>=0 安全
    grad = np.zeros_like(q)
    grad[0] = np.sign(q[0] - q_obs)
    return float(h), grad

ctrl.add_barrier(sphere_barrier, name="obs1")
```

* 若 QP **不可行**：

  1. 縮小 `u_des`（例如 ×0.8）或放鬆 bound slack；
  2. 仍不可行則裁剪 `u_des`；
  3. 適度調整 `max_joint_vel`、`dt`、`qp_slack_weight_*`。

### 選項（OptionSpec）/ Macro-actions

* 內建字串會自動轉為 `OptionSpec`：

  * `hold` → `goal_offset_scale=0.0`
  * `dmp_small_step` / `medium` / `large` → 0.25 / 0.5 / 1.0
* 你也可以自行傳入 `OptionSpec`：

```python
options = [
  OptionSpec("hold", goal_offset_scale=0.0),
  OptionSpec("reach_quarter", goal_offset_scale=0.25, duration_scale=1.2),
]
cfg = HRLConfig(..., options=options)
```

### ROS 2 線上控制（選用） / ROS 2 Online Control (Optional)

```python
ctrl.run_ros2_closed_loop(
    controller_ns="/joint_trajectory_controller",
    hz=10.0,
    topic_joint_states="/joint_states",
    topic_joint_traj="/joint_trajectory",
    frame_id="base_link",
    goal=None,  # or np.array([...])
)
```

* 訂閱 `~/set_goal`（`std_msgs/Float64MultiArray`）可**即時更改目標**。
* 請確認 `joint_names` 的順序在整個流程一致。

### 儲存/載入 / Save & Load

```python
ctrl.save_policy("checkpoint.pt")
# ... later ...
ctrl.load_policy("checkpoint.pt")
```

會保存：policy/target、optimizer、`HRLConfig`、DMP 權重與基底、狀態正規化統計、OptionSpec 列表。

### 常見疑難排解 / Troubleshooting

* **QP 不可行**：降低 `dt`、減小 `max_joint_vel`、開啟/增大 slack；檢查 barrier 梯度方向是否正確（遠離危險為正）。
* **學習慢或不穩**：

  * 確認 `state_dim` 設對（是否含 `2*n_joints` 的增廣部分）。
  * 開啟 `double_dqn`；使用 `target_update="soft"`、`tau=0.01`。
  * 檢查 reward 與觀測尺度，開啟 `state_norm`。
* **ROS 訊息不同步**：確保 `joint_names` 順序一致、`time_from_start` 單調遞增、控制頻率與控制器匹配。
* **相依套件**：若無 OSQP 會退到 `qpsolvers`；未安裝 ROS 仍可 `import`，但呼叫 ROS 函式會拋出明確錯誤。

### 推薦設定 / Suggested Defaults

* `double_dqn=True`，`target_update="soft"`，`tau=0.01`。
* `state_norm=True`，`goal_in_state=True`。
* `dmp_n_basis≈15`，`dmp_alpha_z=25`，`dmp_beta_z=6.25`。
* `qp_R_diag=1.0`，`qp_slack_weight_cbf=1e3`，`qp_slack_weight_bounds=1e2`。

---
