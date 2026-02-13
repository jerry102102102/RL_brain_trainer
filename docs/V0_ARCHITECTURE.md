# RL Brain Trainer v0 Architecture (Robot Learning Final Project)

## Purpose
Turn the existing HRL stack into a **final-project-ready, testable architecture** with clear ablations and measurable convergence behavior.

## 3-Layer Design (aligned with your latest direction)

### 1) Strategic Layer (pre-trained planner)
- Role: task decomposition / sub-goal proposal
- Input: task instruction, environment summary (and optional visual features)
- Output: `task_id`, `subgoal`, `termination condition`
- Training policy: **no heavy training in v0** (use pre-trained/frozen module)

### 2) Tactical Layer (main trainable layer)
- Role: sequence-level decision + adaptation
- Candidate design: memory module (LSTM/GRU) + policy head
- Input: task token + observation history window
- Output: desired trajectory token or waypoint stream (`q_desired` / latent skill code)
- Training policy: **primary optimization target in v0**

### 3) Execution Layer (stable controller)
- Role: robust low-level control under constraints
- Candidate: PID baseline + (optional) RBF/LWR compensation
- Input: desired trajectory and current joint state
- Output: control command `u` / `tau`
- Training policy: mostly analytical/tuned controller (not end-to-end RL in v0)

---

## v0 Research Questions
1. Does adding a memory-based tactical layer improve convergence speed and stability vs baseline?
2. Under disturbance/noise, does the 3-layer design recover faster than single-policy control?
3. Can we reduce control effort while keeping comparable tracking quality?

## Minimal Scenario (start simple)
- Task: fixed target reaching / short-horizon trajectory tracking
- Disturbances: random force pulse OR sensor noise OR friction change
- Environment: current Gazebo setup in `hrl_ws`

## Metrics (must report)
- Success rate
- Time-to-convergence
- Tracking RMSE
- Disturbance recovery time
- Control effort (e.g., ||u|| integral)

## v0 Ablation Plan
- A0: Existing baseline controller/policy (current repo path)
- A1: + Tactical memory module
- A2: + Execution compensation (RBF/LWR) on top of A1

## Deliverables
1. Architecture diagram + dataflow
2. Reproducible config for A0/A1/A2
3. Comparison table + 2-3 plots
4. Short final-project narrative (problem → method → results → limits)

## 7-day Sprint (practical)
- Day 1: freeze interfaces + scenario + metrics
- Day 2-3: baseline run + logging cleanup
- Day 4-5: tactical memory integration
- Day 6: execution compensation + ablation run
- Day 7: plots + writeup
