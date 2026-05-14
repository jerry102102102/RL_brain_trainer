# Final Project Summary

Purpose: A concise landing page for the final ENPM690 project package.

## Project Title

Modular Three-Layer RL for Kitchen Manipulation

## One-Sentence Outcome

This project demonstrates a modular L1-to-RL manipulation prototype with a working kinematic RL skill stack, a Qwen semantic bridge, and a route-curriculum extension from local correction to long-prefix route following.

## What Is Completed

- A pure kinematic Gymnasium environment for a 7-DoF arm.
- PPO/TD3 training and deterministic evaluation infrastructure.
- A simplified and validated `Approach -> Finisher` skill stack.
- A Qwen MCP bridge that converts semantic commands into safe structured skill requests.
- A route curriculum system using dense q-goal targets as policy observations.
- A route-following improvement from longest prefix 21 to full-route probe prefix 170.
- Workspace expansion experiments that push the home-start policy beyond Stage 5.
- A random-start / mixed-start coverage experiment reaching about 80.2% success inside the known workspace.
- A real Gazebo / ROS2 controlled-sim evidence run with camera video and action-server execution logs.

## What Is Not Completed

- Full holder1 -> holder8 transport.
- Full continuous reachable workspace coverage.
- Full Gazebo physics validation for the complete holder1 -> holder8 transport route.
- Real camera grounding.
- Contact, friction, and object interaction.
- Real robot deployment.

## Official Demos

- Demo 1: Qwen L1 bridge.
- Demo 2: kinematic `Approach -> Finisher` skill stack.
- Demo 3: route curriculum extension.
- Demo 4: Gazebo/RViz live recording path for L1 -> L2 -> L3 visualization and local learned-policy motion.

## Final Package Files

- `report/FINAL_REPORT.md`
- `report/FINAL_REPORT.pdf`
- `report/FINAL_PRESENTATION.pptx`
- `report/DEMO_VIDEO_SCRIPT.md`
- `report/REAL_GZ_DEMO_EVIDENCE.md`
- `docs/CURRENT_IMPLEMENTATION.md`
- `report/OFFICIAL_ARTIFACTS.md`
