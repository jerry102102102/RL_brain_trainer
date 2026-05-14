# Final Codes Docker Submission

Project: **Robot Brain Trainer - Modular L1-to-RL Manipulation Stack**

This package is the minimal submission wrapper for the final code demo. It does not retrain policies. It packages the existing final demo code so an evaluator can run:

1. **Local test demo**: Approach -> Finisher local manipulation skill.
2. **Full route / route-prefix demo**: route-curriculum evidence for dense route following.

The repository for the project is:

```text
https://github.com/jerry102102102/Robot_brain_trainer
```

## What This Demonstrates

- A Qwen/L1 semantic bridge that resolves a natural-language tray-moving command into a structured task intent.
- A learned RL skill stack using the final simplified path: `Approach -> Finisher`.
- Headless kinematic evaluation for reproducible code checks.
- Native Gazebo/RViz scripts for visual screen recording.
- Route-curriculum evidence showing route-prefix following, without claiming full holder1-to-holder8 transport is solved.

## What This Does Not Claim

- It does not claim complete kitchen manipulation is solved.
- It does not claim full holder1-to-holder8 tray transport is solved.
- It does not claim real robot deployment.
- Docker headless mode does not provide Gazebo GUI rendering.

## Recommended Paths

| Mode | Recommended for | Command family |
|---|---|---|
| Docker headless | Reproducible code/package check | `docker compose ... run demo ...` |
| Native WSL2/ROS2/Gazebo | Screen recording with GUI | `scripts/final/run_final_*_demo.sh` |

Docker headless is the safest reproducible path. Gazebo GUI may require native ROS2/Gazebo, X11, GPU, and host display support; the native scripts are the intended visual recording path.

## Quick Start With Docker

```bash
git clone https://github.com/jerry102102102/Robot_brain_trainer.git
cd Robot_brain_trainer

docker compose -f final_codes_docker/docker-compose.demo.yaml build

docker compose -f final_codes_docker/docker-compose.demo.yaml run --rm demo \
  bash final_codes_docker/run_dry_check.sh

docker compose -f final_codes_docker/docker-compose.demo.yaml run --rm demo \
  bash final_codes_docker/download_demo_assets.sh

docker compose -f final_codes_docker/docker-compose.demo.yaml run --rm demo \
  bash final_codes_docker/run_local_test_demo.sh

docker compose -f final_codes_docker/docker-compose.demo.yaml run --rm demo \
  bash final_codes_docker/run_full_route_demo.sh
```

The Docker route command defaults to a CPU-stable 90-waypoint route-prefix check. To run the longer prefix120 check:

```bash
docker compose -f final_codes_docker/docker-compose.demo.yaml run --rm \
  -e FULL_ROUTE_END_INDEX=120 demo bash final_codes_docker/run_full_route_demo.sh
```

If trained model files are not present, `download_demo_assets.sh` prints exactly which files are missing and where they should be placed. If a URL is added to `model_manifest.yaml`, the script can download it.

## Quick Start Without Docker

From the repo root:

```bash
bash scripts/final/check_final_demo_ready.sh
bash scripts/final/run_final_local_test_demo.sh
bash scripts/final/run_final_full_route_demo.sh
```

For visual Gazebo screen recording, first launch the scene natively:

```bash
source /opt/ros/jazzy/setup.zsh
source external/ENPM662_Group4_FinalProject/install/setup.zsh
ros2 launch kitchen_robot_description gazebo.launch.py use_sim_time:=true headless:=false
```

Then in another terminal:

```bash
bash scripts/final/run_live_gz_screen_recording_demo.sh local_skill
bash scripts/final/run_live_gz_screen_recording_demo.sh tray_like_transport
```

## Trained Model Assets

Required model paths are listed in:

```text
final_codes_docker/model_manifest.yaml
```

The selected model files are small SB3 `.zip` policies. If they are not bundled with the submission, place them at the expected paths listed in the manifest, or add download URLs to the manifest before running `download_demo_assets.sh`.

## Demo Videos

Provided reference videos:

| Demo | Link |
|---|---|
| Local test | `https://drive.google.com/file/d/1_faSuh4cMAKPs8mhAdgD_rcqrU7K045F/view?usp=drive_link` |
| Full route / route-prefix | `https://drive.google.com/file/d/125XAsfr5hSjJBB1QiJ0ROllNVFENyp7f/view?usp=drive_link` |

These videos show the expected visual result. The code package provides the scripts to reproduce the same kinds of demos.

## Expected Outputs

| Demo | Output directory |
|---|---|
| Local test | `report/demo_outputs/final_local_test_*` |
| Full route / route-prefix | `report/demo_outputs/final_full_route_*` |
| Asset check | `final_codes_docker/asset_check/` |

Each demo writes a `command_log.txt` plus a JSON summary.

## Known Limitations

- Docker GUI Gazebo is optional and environment-dependent.
- Headless Docker mode validates Python package/eval paths, not the rendered Gazebo scene.
- Docker uses CPU inference by default. Long sequential route rollouts can show small accumulated numerical differences from the native CUDA/venv reference path. The default Docker route-prefix length is therefore set to 90 for robust reproduction; set `FULL_ROUTE_END_INDEX=120` for the longer reference check.
- The full route demo is route-curriculum evidence; it must not be described as full physical tray transport completion.
- If trained models are absent and no URLs are provided, scripts fall back to summary artifacts only for dry demonstration and clearly report that policy execution was skipped.
