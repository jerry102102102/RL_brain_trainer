from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any

from .common import add_common_io_args, finalize_output, load_yaml, maybe_load_ros, stamp_to_ns, tool_result
from .metrics_core import summarize_image_health


@dataclass
class ImageSample:
    recv_ns: int
    header_ns: int | None
    width: int | None = None
    height: int | None = None
    encoding: str | None = None


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
            self.set_parameters([
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, bool(wp0.get("use_sim_time", False))
                )
            ])
            self.samples: dict[str, list[ImageSample]] = defaultdict(list)
            self.camera_info_seen: dict[str, int] = defaultdict(int)
            self.image_contract_obs: dict[str, dict[str, Any]] = {}
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
                ros_recv_ns = self.get_clock().now().nanoseconds
                wall_recv_ns = time.time_ns()
                header_ns = None
                try:
                    header_ns = stamp_to_ns(msg.header.stamp)
                except Exception:
                    header_ns = None
                if header_ns is not None:
                    recv_ns = ros_recv_ns if abs(ros_recv_ns - header_ns) <= abs(wall_recv_ns - header_ns) else wall_recv_ns
                else:
                    recv_ns = ros_recv_ns
                width = None
                height = None
                encoding = None
                try:
                    width = int(getattr(msg, "width"))
                    height = int(getattr(msg, "height"))
                    encoding = str(getattr(msg, "encoding"))
                except Exception:
                    pass
                if topic not in self.image_contract_obs:
                    self.image_contract_obs[topic] = {
                        "message_type": "sensor_msgs/msg/Image",
                        "width": width,
                        "height": height,
                        "encoding": encoding,
                    }
                self.samples[topic].append(
                    ImageSample(
                        recv_ns=recv_ns,
                        header_ns=header_ns,
                        width=width,
                        height=height,
                        encoding=encoding,
                    )
                )

            return cb

        def _info_cb(self, topic: str):
            def cb(_msg: Any) -> None:
                self.camera_info_seen[topic] += 1

            return cb

    rclpy.init()
    node = ImgDiagNode()
    try:
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)

        per_topic: dict[str, Any] = {}
        topic_to_ok: list[bool] = []
        for cam_name, cam in image_cfgs.items():
            topic = cam["image_topic"]
            samples = node.samples.get(topic, [])
            recv = [s.recv_ns for s in samples]
            header = [s.header_ns for s in samples]

            # Replay path: normalize to relative timeline to avoid absolute epoch offsets
            # (record-time header stamp vs replay-time receive stamp) inflating latency.
            latency_header = header
            latency_basis = "absolute_recv_minus_header"
            recv_non_none = [x for x in recv if x is not None]
            header_non_none = [x for x in header if x is not None]
            if args.replay_mode and recv_non_none and header_non_none:
                recv0 = int(recv_non_none[0])
                header0 = int(header_non_none[0])
                latency_header = [None if h is None else int(h) - header0 for h in header]
                recv = [int(r) - recv0 for r in recv]
                latency_basis = "relative_timeline_recv_minus_header"

            recv_eval = recv
            header_eval = latency_header
            dropped_mixed = 0
            if args.replay_mode:
                filt_recv: list[int] = []
                filt_header: list[int | None] = []
                for r, h in zip(recv, latency_header):
                    if h is None:
                        continue
                    d_ms = (int(r) - int(h)) / 1e6
                    # Replay-only filter: reject mixed live stream samples with huge offset.
                    if abs(d_ms) <= 5000.0:
                        filt_recv.append(int(r))
                        filt_header.append(int(h))
                    else:
                        dropped_mixed += 1
                if filt_recv:
                    recv_eval = filt_recv
                    header_eval = filt_header

            metrics = summarize_image_health(
                recv_stamps_ns=recv_eval,
                header_stamps_ns=header_eval,
                expected_fps=float(cam.get("expected_fps", 0.0) or 0.0),
                latency_p95_limit_ms=latency_limit_ms,
            )
            metrics["latency_basis"] = latency_basis
            if args.replay_mode:
                metrics["dropped_mixed_stream_samples"] = dropped_mixed
            metrics["camera_info_topic"] = cam.get("camera_info_topic")
            if cam.get("camera_info_topic"):
                metrics["camera_info_seen"] = int(node.camera_info_seen.get(cam["camera_info_topic"], 0))
            metrics["observed_contract"] = node.image_contract_obs.get(topic)
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
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
