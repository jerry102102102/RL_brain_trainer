from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .common import add_common_io_args, finalize_output, load_yaml, maybe_load_ros, stamp_to_ns, tool_result
from .metrics_core import summarize_image_health


@dataclass
class ImageSample:
    recv_ns: int
    header_ns: int | None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 image health diagnostics (fps/drop/latency).")
    add_common_io_args(p)
    p.add_argument("--duration-sec", type=float, default=None)
    p.add_argument("--replay-mode", action="store_true", help="Use replay image latency gate (P95<120ms)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]
    thresholds = wp0["thresholds"]
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    image_cfgs = wp0.get("cameras", {})
    latency_limit_ms = float(
        thresholds["replay_image_latency_p95_ms"] if args.replay_mode else thresholds["image_latency_p95_ms"]
    )

    rclpy, Node = maybe_load_ros()

    class ImgDiagNode(Node):
        def __init__(self) -> None:
            super().__init__("v5_wp0_image_health_diag")
            self.samples: dict[str, list[ImageSample]] = defaultdict(list)
            self.camera_info_seen: dict[str, int] = defaultdict(int)
            self.subs = []
            for cam_name, cam in image_cfgs.items():
                img_topic = cam["image_topic"]
                info_topic = cam.get("camera_info_topic")
                from sensor_msgs.msg import Image, CameraInfo  # type: ignore

                self.subs.append(self.create_subscription(Image, img_topic, self._img_cb(img_topic), 10))
                if info_topic:
                    self.subs.append(self.create_subscription(CameraInfo, info_topic, self._info_cb(info_topic), 10))

        def _img_cb(self, topic: str):
            def cb(msg: Any) -> None:
                recv_ns = self.get_clock().now().nanoseconds
                header_ns = None
                try:
                    header_ns = stamp_to_ns(msg.header.stamp)
                except Exception:
                    header_ns = None
                self.samples[topic].append(ImageSample(recv_ns=recv_ns, header_ns=header_ns))

            return cb

        def _info_cb(self, topic: str):
            def cb(_msg: Any) -> None:
                self.camera_info_seen[topic] += 1

            return cb

    rclpy.init()
    node = ImgDiagNode()
    try:
        end_time = node.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while rclpy.ok() and node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(node, timeout_sec=0.2)

        per_topic: dict[str, Any] = {}
        topic_to_ok: list[bool] = []
        for cam_name, cam in image_cfgs.items():
            topic = cam["image_topic"]
            samples = node.samples.get(topic, [])
            recv = [s.recv_ns for s in samples]
            header = [s.header_ns for s in samples]
            metrics = summarize_image_health(
                recv_stamps_ns=recv,
                header_stamps_ns=header,
                expected_fps=float(cam.get("expected_fps", 0.0) or 0.0),
                latency_p95_limit_ms=latency_limit_ms,
            )
            metrics["camera_info_topic"] = cam.get("camera_info_topic")
            if cam.get("camera_info_topic"):
                metrics["camera_info_seen"] = int(node.camera_info_seen.get(cam["camera_info_topic"], 0))
            per_topic[topic] = metrics
            gate = metrics.get("latency", {}).get("gate", {})
            topic_to_ok.append(bool(gate.get("pass", False)))

        out = tool_result(
            "image_health_diag",
            cfg,
            {
                "duration_sec": duration_sec,
                "replay_mode": bool(args.replay_mode),
                "latency_gate_p95_ms": latency_limit_ms,
                "per_topic": per_topic,
            },
            ok=all(topic_to_ok) if topic_to_ok else False,
        )
        finalize_output(out, args)
        return 0 if out.get("ok") else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
