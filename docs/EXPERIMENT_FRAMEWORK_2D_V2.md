# Experiment Framework v2: 2D Acceleration + Steering + Realistic Disturbances

## Purpose
Start from a **small but meaningful** benchmark to validate the 3-layer architecture before scaling to full manipulation.

## Task Definition
- State: 2D pose + velocity
  - `s_t = [x, y, yaw, v, omega, goal_x, goal_y]`
- Action:
  - `a_t = [a_linear, a_angular]` (acceleration + steering-rate command)
- Objective:
  - Reach goal with stable heading and low control effort under disturbances.

## Why this scenario first
1. Fast iteration cycle (cheap to run, easy ablation)
2. Cleanly tests layer responsibilities:
   - L1: subgoal/option planning
   - L2: memory-based temporal adaptation
   - L3: robust tracking and disturbance rejection
3. Produces quantitative convergence evidence quickly.

---

## Disturbance Set (realistic)
- D1 Sensor noise: Gaussian + occasional bias on position/yaw
- D2 Actuation delay: fixed/random command latency (e.g., 40–120 ms)
- D3 Friction drift: domain randomization on drag coefficient
- D4 Impulse perturbation: random external push at random timestep
- D5 Partial observation dropout: short missing observations (1–3 frames)

Disturbance levels:
- `easy`, `medium`, `hard` with predefined parameter ranges.

---

## Layer Mapping in this experiment

### L1 Strategic (frozen)
- Input: task text + map/observation summary
- Output: `option_id`, local subgoal `(x*, y*)`, termination condition
- Rule: no direct low-level control output.

### L2 Tactical (trainable, main focus)
- Model: memory + LSTM/GRU
- Input: option + history window
- Output: `q_desired` (or local target trajectory)
- Key metric: sample efficiency + adaptation speed under disturbances

### L3 Execution (stable control)
- Baseline: PID/LQR
- Enhanced: +RBF/LWR compensation
- Advanced: +MRAC-TDE-NN branch
- Key metric: tracking stability and recovery under perturbation

---

## Ablation Plan
- A0: L1 frozen + L2 no-memory + L3 baseline
- A1: A0 + L2 memory module
- A2: A1 + L3 RBF/LWR compensation
- A3: A2 + L3 MRAC-TDE-NN

## Evaluation Matrix
- Environments: E0(no disturbance), E1(sensor), E2(actuation), E3(mixed)
- Task sets: T0(single goal), T1(two-stage goal)
- Seeds: >=5 per config

---

## Metrics (must log)
- Success rate
- Time-to-convergence
- Tracking RMSE
- Disturbance recovery time
- Control effort `∫||u||dt`
- Constraint violation count (if safety bounds enabled)

---

## Reproducibility Contract
Each run must record:
- git commit hash
- config filename
- random seed
- disturbance profile id
- result artifact path

---

## Codex Linkage (Research mode x Codex)
Use Codex for implementation milestones with strict task packs:
1. Plan first, then patch
2. One milestone per run
3. Mandatory test commands + output report
4. Assistant review gate before merge

### Suggested milestone order
- M1: implement env + disturbance wrappers + logger schema
- M2: run A0 baseline and export metrics
- M3: add L2 memory and run A1 comparison
- M4: add L3 compensation branch (A2/A3)

---

## Immediate Next Action (30-60 min)
1. Create `config/exp_2d_a0.yaml` and `config/exp_2d_a1.yaml`
2. Create logger fields for 6 core metrics
3. Run one smoke experiment (E0/T0, seed=0)
