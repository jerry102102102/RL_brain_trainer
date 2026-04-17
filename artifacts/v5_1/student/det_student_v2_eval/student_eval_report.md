# Deterministic Student Evaluation

- student_checkpoint: `artifacts/v5_1/student/det_student_v2/checkpoint_best.pt`
- student_run_id: `det_student_v2`
- fixed_eval_suite_id: `det_eval_near_home_700001_TC0_5`

## Student Metrics
- true_outer_hit_rate: `0.2000`
- true_inner_hit_rate: `0.0000`
- true_dwell_hit_rate: `0.0000`
- true_basin_hit_rate: `0.2000`
- mean_final_dpos: `0.100820`
- regression_rate: `0.8000`
- final_action_l2_mean: `0.006397`

## Teacher Baselines
- `main_reward_bonly_baseline_s0b_tc0_act008_001`: outer=`0.2000`, inner=`0.0000`, mean_final_dpos=`0.098486`, regression=`0.8000`
- `main_reward_bonly_solidify_s0b_tc0_act008_001`: outer=`0.2000`, inner=`0.0000`, mean_final_dpos=`0.102322`, regression=`0.8000`
- `main_reward_bonly_solidify_v2_s0b_tc0_act008_001`: outer=`0.2000`, inner=`0.0000`, mean_final_dpos=`0.101734`, regression=`0.8000`

## Success Criteria
- level1_outer_mean_final: `False`
- level2_inner_nonzero: `False`
- level3_success_higher: `False`

## Verdict
- teacher-student deterministic extraction is better only if outer/inner/final retention improve together.
