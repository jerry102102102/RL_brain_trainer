#!/usr/bin/env python3
"""Record a ROS2 Image topic to MP4 for the real Gazebo demo."""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image


STOP = False


def _handle_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
    global STOP
    STOP = True


def _image_msg_to_pil(msg) -> Image.Image:  # type: ignore[no-untyped-def]
    width = int(msg.width)
    height = int(msg.height)
    encoding = str(msg.encoding).lower()
    data = bytes(msg.data)
    if encoding in ("rgb8", "rgb"):
        return Image.frombytes("RGB", (width, height), data)
    if encoding in ("rgba8", "rgba"):
        return Image.frombytes("RGBA", (width, height), data).convert("RGB")
    if encoding in ("bgr8", "bgr"):
        raw = Image.frombytes("RGB", (width, height), data)
        r, g, b = raw.split()
        return Image.merge("RGB", (b, g, r))
    if encoding in ("bgra8", "bgra"):
        raw = Image.frombytes("RGBA", (width, height), data)
        b, g, r, _a = raw.split()
        return Image.merge("RGB", (r, g, b))
    if encoding in ("mono8", "8uc1"):
        return Image.frombytes("L", (width, height), data).convert("RGB")
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _write_video(frames: list[Image.Image], output: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames captured; cannot write video")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to write MP4")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, frame in enumerate(frames):
            frame.save(tmp_dir / f"frame_{idx:05d}.png")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(tmp_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record ROS2 Image topic to MP4")
    parser.add_argument("--topic", default="/v5/cam/side/rgb")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--warmup-timeout", type=float, default=15.0)
    args = parser.parse_args()

    import json
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ImageMsg

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    rclpy.init(args=None)
    node = rclpy.create_node("final_gz_camera_recorder")
    frames: list[Image.Image] = []
    first_stamp = None
    last_stamp = None
    last_frame_time = 0.0
    frame_period = 1.0 / max(1, int(args.fps))

    def on_image(msg: ImageMsg) -> None:
        nonlocal first_stamp, last_stamp, last_frame_time
        now = time.monotonic()
        if frames and (now - last_frame_time) < frame_period:
            return
        try:
            frame = _image_msg_to_pil(msg)
        except Exception as exc:
            node.get_logger().warn(f"Skipping frame: {exc}")
            return
        frames.append(frame)
        last_frame_time = now
        stamp = int(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        if first_stamp is None:
            first_stamp = stamp
        last_stamp = stamp

    sub = node.create_subscription(ImageMsg, args.topic, on_image, qos_profile_sensor_data)
    del sub

    deadline = time.monotonic() + max(0.1, float(args.duration))
    warmup_deadline = time.monotonic() + max(0.1, float(args.warmup_timeout))
    try:
        while not STOP and time.monotonic() < deadline and len(frames) < max(1, int(args.max_frames)):
            try:
                rclpy.spin_once(node, timeout_sec=0.05)
            except (ExternalShutdownException, KeyboardInterrupt):
                break
            if not frames and time.monotonic() > warmup_deadline:
                break
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

    output = Path(args.output)
    summary = {
        "topic": args.topic,
        "output": str(output),
        "frames_captured": len(frames),
        "fps": int(args.fps),
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "duration_requested_s": float(args.duration),
    }
    if frames:
        _write_video(frames, output, args.fps)
        summary["video_written"] = True
        summary["width"] = frames[0].width
        summary["height"] = frames[0].height
    else:
        summary["video_written"] = False

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
