# RL-brainer V3 — Online Memory + LSTM (Research-mode plan)

## Goal
Replace offline imitation flow with online interaction learning so memory and LSTM affect decisions through reward-driven updates.

## Causal hypothesis
- Offline supervised loop cannot fully optimize task success under shifted rollout distribution.
- Online loop (act->step->store->update) will reduce far-timeout by learning from encountered failures.

## V3 architecture
1. High-level planner (heuristic subgoal/option) kept fixed for now.
2. Tactical policy: recurrent actor-critic (LSTM backbone).
3. Episodic memory: kNN retrieval with confidence gating; retrieval used as auxiliary context.
4. Low-level controller: fixed RBF execution (hold constant to isolate tactical effect).

## Training loop (online)
For each step:
1) policy produces action with LSTM hidden state
2) env step returns transition
3) push transition to replay (and episodic memory)
4) every N steps: update actor/critic weights
5) periodically evaluate deterministic policy

## Required diagnostics
- success_rate (mean/std over seeds)
- done_reasons and timeout distance bins (near/mid/far)
- learning curve (episode return vs steps)
- enter_dock_zone_rate
- memory retrieval confidence stats

## Ablation set
- B: online recurrent RL without memory retrieval
- C: online recurrent RL + memory retrieval
(Compare both against prior offline baseline)

## Immediate implementation order
1. Add new trainer script `train_rl_brainer_v3_online.py`
2. Implement recurrent actor-critic update loop (minimal stable version)
3. Add episodic memory module and gated retrieval
4. Add logging JSON artifact for diagnostics
5. Run quick B/C experiments (same seeds/config)
