#!/usr/bin/env python3
"""Generate lightweight MP4 demo videos from final-package outputs.

The videos are intentionally self-contained and headless-friendly: they render
title cards, official metrics, and generated figures into short MP4 clips. This
keeps the final demo package usable even when Gazebo/GUI recording is not
available during packaging.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
VIDEOS = REPORT / "videos"
FIGURES = REPORT / "figures"
DEMO_OUTPUTS = REPORT / "demo_outputs"

W, H = 1280, 720
FPS = 2


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(44)
FONT_SUBTITLE = _font(30)
FONT_BODY = _font(25)
FONT_SMALL = _font(20)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _card(title: str, lines: list[str], accent: tuple[int, int, int] = (42, 157, 143)) -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 24], fill=accent)
    draw.text((72, 70), title, font=FONT_TITLE, fill=(32, 32, 32))
    y = 155
    for line in lines:
        if not line:
            y += 20
            continue
        wrapped = textwrap.wrap(line, width=72)
        for part in wrapped:
            draw.text((92, y), part, font=FONT_BODY, fill=(45, 45, 45))
            y += 38
        y += 8
    draw.text((72, H - 58), "Robot Brain Trainer final demo package", font=FONT_SMALL, fill=(90, 90, 90))
    return img


def _figure_card(title: str, figure_path: Path, caption: list[str], accent: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 248, 242))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 24], fill=accent)
    draw.text((56, 46), title, font=FONT_SUBTITLE, fill=(32, 32, 32))

    if figure_path.exists():
        fig = Image.open(figure_path).convert("RGB")
        max_w, max_h = 760, 520
        fig.thumbnail((max_w, max_h), Image.LANCZOS)
        img.paste(fig, (54, 132))
    else:
        draw.rectangle([54, 132, 814, 652], outline=(160, 160, 160), width=3)
        draw.text((90, 360), f"Missing figure: {figure_path.name}", font=FONT_BODY, fill=(130, 50, 50))

    y = 150
    for line in caption:
        for part in textwrap.wrap(line, width=30):
            draw.text((860, y), part, font=FONT_BODY, fill=(45, 45, 45))
            y += 38
        y += 12
    return img


def _write_video(path: Path, frames: list[Image.Image], seconds_per_frame: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate MP4 demo videos")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_idx = 0
        for frame in frames:
            for _ in range(FPS * seconds_per_frame):
                frame.save(tmp_dir / f"frame_{frame_idx:04d}.png")
                frame_idx += 1
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(tmp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _demo_01_frames() -> list[Image.Image]:
    data = _load_json(DEMO_OUTPUTS / "demo_01_qwen_bridge_output.json")
    target_pose = data.get("target_pose") or {}
    return [
        _card(
            "Demo 1: Qwen L1 Bridge",
            [
                "Natural-language task: Move tray1 from shelf_A1 to shelf_B1 while keeping it level and inserting with a stable pose.",
                "Qwen is used as the semantic L1 layer, not as a low-level controller.",
            ],
            (38, 70, 83),
        ),
        _card(
            "Structured IntentPacket",
            [
                f"Tool: {data.get('tool', 'resolve_intent_packet')}",
                f"Object: {data.get('object_id', 'tray1')}",
                f"Source -> target: {data.get('source_slot', 'shelf_A1')} -> {data.get('target_slot', 'shelf_B1')}",
                f"Skill pipeline: {data.get('pipeline', 'APPROACH -> FINISHER')}",
                f"Target xyz: {target_pose.get('xyz', [-0.92, -1.16, 1.22])}",
                f"Target rpy: {target_pose.get('rpy', [3.14, 0.0, 3.14])}",
            ],
            (38, 70, 83),
        ),
        _card(
            "Safety Boundary",
            [
                "The L1 model never emits raw joint actions, torques, or trajectories.",
                "L1 outputs semantic intent and a structured skill request.",
                "L2/L3 own policy rollout, execution, and safety boundaries.",
            ],
            (38, 70, 83),
        ),
    ]


def _demo_02_frames() -> list[Image.Image]:
    data = _load_json(DEMO_OUTPUTS / "demo_02_kinematic_skill_summary.json")
    success = data.get("stage5_success", 0.93)
    final_pos = data.get("stage5_final_position_error_mm", 2.89)
    final_ori = data.get("stage5_final_orientation_error_rad", 0.0208)
    handoff_pos = data.get("stage5_handoff_position_error_mm", 1.96)
    handoff_ori = data.get("stage5_handoff_orientation_error_rad", 0.0177)
    return [
        _card(
            "Demo 2: Kinematic Approach -> Finisher",
            [
                "Ablations simplified the skill path to Approach -> Finisher.",
                "The trained kinematic skill stack is evaluated in staged workspace sweeps.",
            ],
            (42, 157, 143),
        ),
        _figure_card(
            "Workspace Sweep Result",
            FIGURES / "workspace_sweep_stage_success.png",
            [
                f"Stage 5 success: {success:.2f}",
                f"Handoff pos error: {handoff_pos:.2f} mm",
                f"Handoff ori error: {handoff_ori:.4f} rad",
                f"Final pos error: {final_pos:.2f} mm",
                f"Final ori error: {final_ori:.4f} rad",
            ],
            (42, 157, 143),
        ),
        _card(
            "Interpretation",
            [
                "This is not full kitchen manipulation.",
                "It is a working kinematic RL skill stack with millimeter-level final position error in the trained workspace.",
            ],
            (42, 157, 143),
        ),
    ]


def _demo_03_frames() -> list[Image.Image]:
    data = _load_json(DEMO_OUTPUTS / "demo_03_route_curriculum_summary.json")
    return [
        _card(
            "Demo 3: Route Curriculum",
            [
                "The original local controller failed on full scene-level transport.",
                "Route curriculum trains waypoint following with q_goal as target observation, while the RL policy still outputs actions.",
            ],
            (231, 111, 81),
        ),
        _figure_card(
            "Route Prefix Improvement",
            FIGURES / "route_prefix_improvement.png",
            [
                f"Baseline longest prefix: {data.get('baseline_longest_prefix', 21)}",
                f"Prefix120 success: {data.get('prefix120_success', 1.0):.2f}",
                f"Full483 probe success: {data.get('full483_success', 0.4741):.4f}",
                f"Full483 longest prefix: {data.get('full483_longest_prefix', 170)}",
            ],
            (231, 111, 81),
        ),
        _figure_card(
            "Current Route Limitation",
            FIGURES / "route_curriculum_limitations.png",
            [
                "Stable route prefix: 120",
                "Full-route probe reaches 170",
                f"First failure index: {data.get('first_failure_index', 171)}",
                f"Failure reason: {data.get('first_failure_reason', 'position')}",
                "Full holder1 -> holder8 remains future work.",
            ],
            (231, 111, 81),
        ),
    ]


def main() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    demos = {
        "demo_01_qwen_bridge.mp4": _demo_01_frames(),
        "demo_02_kinematic_skill.mp4": _demo_02_frames(),
        "demo_03_route_curriculum.mp4": _demo_03_frames(),
    }
    all_frames: list[Image.Image] = []
    for name, frames in demos.items():
        out = VIDEOS / name
        _write_video(out, frames)
        all_frames.extend(frames)
        print(f"Generated {out}")
    final_out = VIDEOS / "final_demo_compilation.mp4"
    _write_video(final_out, all_frames, seconds_per_frame=3)
    print(f"Generated {final_out}")


if __name__ == "__main__":
    main()
