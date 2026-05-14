# Final Presentation Slide Source

1. Title: Modular L1-to-RL Manipulation Stack with Route-Curriculum Extension.
2. Problem Setup: kitchen manipulation needs semantic interpretation, learned control, and safe execution.
3. L1/L2/L3 Architecture: LLM/VLM never outputs raw joint commands.
4. Phase 1 Baseline: infrastructure worked; integrated behavior did not.
5. Phase 2 Simplification: diagnostics led to Approach -> Finisher.
6. Kinematic Skill Result: Stage 5 success 0.93, final position 2.89 mm, final orientation 0.0208 rad.
7. Qwen L1 Bridge: natural language to IntentPacket to dry-run skill request.
8. Route Curriculum Motivation: local policy worked; full route failed at prefix 21.
9. Route Curriculum Result: prefix120 success 1.0; full probe reaches prefix 170.
10. What Still Fails: full483 unsolved, prefix180 forgetting, need gated checkpoint selection.
11. Demo Structure: Qwen bridge, kinematic skill, route curriculum.
12. Conclusion: modular stack demonstrated; full Gazebo kitchen transport remains future work.
