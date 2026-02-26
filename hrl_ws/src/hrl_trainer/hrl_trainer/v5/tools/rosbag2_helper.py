from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .common import add_common_io_args, finalize_output, load_yaml, tool_result


def _collect_topics(wp0: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for cam in wp0.get("cameras", {}).values():
        for key in ("image_topic", "camera_info_topic"):
            topic = cam.get(key)
            if topic:
                topics.append(topic)
    for spec in wp0.get("state_topics", []):
        if spec.get("topic"):
            topics.append(spec["topic"])
    for sec in ("pose_jitter",):
        topic = wp0.get(sec, {}).get("topic")
        if topic:
            topics.append(topic)
    id_topic = wp0.get("id_switch", {}).get("topic")
    if id_topic:
        topics.append(id_topic)
    seen = set()
    out: list[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _record_cmd(wp0: dict[str, Any], output_bag: str) -> list[str]:
    return ["ros2", "bag", "record", "--use-sim-time", "-o", output_bag, *_collect_topics(wp0)]


def _replay_cmd(bag_path: str) -> list[str]:
    return ["ros2", "bag", "play", bag_path, "--clock"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 rosbag2 record/replay helper.")
    add_common_io_args(p)
    sp = p.add_subparsers(dest="command", required=True)

    rec = sp.add_parser("record", help="Print or execute ros2 bag record command")
    rec.add_argument("--bag", default="/tmp/v5_wp0_capture")
    rec.add_argument("--execute", action="store_true")

    rep = sp.add_parser("replay", help="Print or execute ros2 bag play command")
    rep.add_argument("bag_path")
    rep.add_argument("--execute", action="store_true")

    sp.add_parser("print-commands", help="Print both record and replay examples")
    return p


def _run_if_requested(cmd: list[str], execute: bool) -> dict[str, Any]:
    res = {"command": cmd, "shell": shlex.join(cmd), "executed": execute}
    if execute:
        cp = subprocess.run(cmd, check=False)
        res["returncode"] = cp.returncode
        res["success"] = cp.returncode == 0
    return res


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    wp0 = cfg["wp0"]

    if args.command == "record":
        rec = _record_cmd(wp0, args.bag)
        metrics = _run_if_requested(rec, args.execute)
        ok = metrics.get("success", True)
    elif args.command == "replay":
        rep = _replay_cmd(args.bag_path)
        metrics = _run_if_requested(rep, args.execute)
        ok = metrics.get("success", True)
    else:
        rec = _record_cmd(wp0, "/tmp/v5_wp0_capture")
        rep = _replay_cmd("/tmp/v5_wp0_capture")
        metrics = {
            "record": {"command": rec, "shell": shlex.join(rec)},
            "replay": {"command": rep, "shell": shlex.join(rep)},
            "notes": [
                "Run record in terminal A during live scene.",
                "Run replay after stopping live publishers to validate topic structure and replay latency.",
            ],
        }
        ok = True

    out = tool_result("rosbag2_helper", cfg, metrics, ok=ok)
    finalize_output(out, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
