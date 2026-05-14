# Full Workspace Coverage Curriculum: Random-Start / Mixed-Start Training

## Why This Experiment Exists

The current Approach -> Finisher policy is a strong local kinematic skill, but it is not yet a full workspace controller. Stage 0-7 are relatively stable, Stage 8+ is partial/stress coverage, and Stage 10/11 are still manually defined difficulty shells rather than the full reachable workspace.

The next experiment changes the training question from:

> Can the policy solve a bigger home-start stage?

to:

> Can the policy start from many valid workspace states and move to many reachable targets?

This matters because a real manipulation system cannot assume every skill begins from home. After one motion finishes, the next skill starts from wherever the arm actually ended.

## Core Principle

Stage ID is not workspace coverage.

Stage 0-11 are evaluation shells. The full reachable workspace is continuous, so this experiment approximates it using sampled target maps, start-state maps, start-target pair buckets, and coverage metrics.

## New Components

- `workspace_target_map.py`: generates a reachable target pool with q target, FK pose, stage shell, bucket ID, joint-limit margin, and difficulty score.
- `workspace_start_state_map.py`: generates non-home start states from home, old successful stage goals, random valid q states, near-target states, and synthetic rollout-like states.
- `start_target_pair_sampler.py`: classifies start-target pairs into retention, local, medium, frontier, and stress pairs.
- `adaptive_frontier_sampler.py`: classifies buckets as mastered, frontier, hard-but-promising, forgetting-risk, stress, or too-hard.
- `eval_full_workspace_coverage.py`: evaluates home-start retention, random-start known workspace, random-start frontier, and full reachable stress splits.

## Training Strategy

The overnight run uses the existing PPO workspace expansion entrypoint, but with random-start pair sampling enabled inside the reset sampler.

Initial mixed-start ratios:

| Source | Ratio | Purpose |
|---|---:|---|
| home start | 15% | keep original behavior alive |
| old successful starts | 25% | replay known workspace |
| random valid q starts | 25% | break home-start dependency |
| frontier pairs | 20% | push Stage 8-11 boundary |
| failure recovery starts | 10% | learn recovery/settling |
| stress starts | 5% | small exposure to harder regions |

The run starts from:

`artifacts/kinematic_phase1/workspace_expansion/workspace_expand_stage10_11_ppo_1h_002/best_checkpoint/model_best_by_gate.zip`

Finisher remains frozen:

`artifacts/kinematic_phase1/phase1c/dock_workspace_handoff_noop_ft_1m_001/model_latest.zip`

## Evaluation Splits

1. **Home-start Stage Eval**
   - Same style as prior Stage 0-11 eval.
   - Purpose: make sure Stage 0-7 retention does not collapse.

2. **Random-start Known Workspace Eval**
   - Starts sampled from Stage 0-7 style states.
   - Targets sampled from Stage 0-8.
   - Purpose: verify the policy can operate without returning home.

3. **Random-start Frontier Eval**
   - Starts from known/partial workspace.
   - Targets from Stage 8-11 frontier buckets.
   - Purpose: measure expansion beyond the stable region.

4. **Full Reachable Stress Eval**
   - Starts and targets sampled broadly from the target/start maps.
   - Purpose: draw the coverage boundary, not claim full success.

## Main Metrics

- Stage 0-7 retention
- random-start known workspace success
- frontier random-start success
- full reachable stress success
- covered bucket fraction
- stable / partial / stress bucket fractions
- max successful joint L2 distance
- average successful EE start-target distance
- failure reason by bucket

## Success Criteria

Minimum useful result:

- Stage 0-7 retention stays healthy.
- Random-start known workspace success is clearly above a home-only baseline.
- Coverage/failure maps are produced.

Good result:

- Random-start known workspace success >= 0.70.
- Frontier random-start success improves.
- Stage 8/9 do not regress.

Strong result:

- Random-start known workspace success >= 0.80.
- Frontier random-start success >= 0.50.
- Coverage cloud visibly expands.

## What This Experiment Will Not Claim

This experiment will not claim:

- Stage 11 is the full workspace.
- full holder1 -> holder8 transport is solved.
- Gazebo physics transport is solved.
- the latest checkpoint is automatically best.

The final promoted model must be selected by retention + random-start + coverage gates, not by the latest file.
