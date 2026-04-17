from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .common import write_json
except ImportError:  # pragma: no cover - script execution fallback
    from hrl_trainer.v5.tools.common import write_json


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[6]

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
REQUIRED_TOPICS = [
    "/clock",
    "/joint_states",
    "/v5/cam/overhead/rgb",
    "/v5/cam/side/rgb",
    "/tray1/pose",
    "/v5/perception/object_pose_est",
]


@dataclass(frozen=True)
class TopicProbeResult:
    topic: str
    listed: bool
    sample: bool
    elapsed_sec: float


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run(cmd: list[str], *, timeout_sec: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)


def _list_topics() -> set[str]:
    cp = _run(["ros2", "topic", "list"], timeout_sec=3.0)
    if cp.returncode != 0:
        return set()
    return {line.strip() for line in cp.stdout.splitlines() if line.strip()}


def _wait_for_topic(topic: str, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if topic in _list_topics():
            return True
        time.sleep(1.0)
    return False


def _wait_for_sample(topic: str, timeout_sec: float) -> bool:
    cp = _run(["ros2", "topic", "echo", topic, "--once"], timeout_sec=timeout_sec)
    return cp.returncode == 0


def _probe_topics(timeout_sec: float) -> list[TopicProbeResult]:
    rows: list[TopicProbeResult] = []
    for topic in REQUIRED_TOPICS:
        started = time.monotonic()
        listed = _wait_for_topic(topic, timeout_sec)
        sample = _wait_for_sample(topic, timeout_sec) if listed else False
        rows.append(
            TopicProbeResult(
                topic=topic,
                listed=listed,
                sample=sample,
                elapsed_sec=round(time.monotonic() - started, 3),
            )
        )
    return rows


def _path_status(rows: list[TopicProbeResult], blocked_reason: str | None = None) -> str:
    if blocked_reason:
        return STATUS_BLOCKED
    if not rows:
        return STATUS_BLOCKED
    return STATUS_PASS if all(row.listed and row.sample for row in rows) else STATUS_FAIL


def _rows_json(rows: list[TopicProbeResult]) -> list[dict[str, Any]]:
    return [
        {
            "topic": row.topic,
            "listed": row.listed,
            "sample": row.sample,
            "elapsed_sec": row.elapsed_sec,
        }
        for row in rows
    ]


def _blocked_path(reason: str) -> dict[str, Any]:
    return {
        "status": STATUS_BLOCKED,
        "blocked_reason": reason,
        "tray_pose_mode": "unknown",
        "topics": _rows_json([]),
    }


def _parse_launch_cmd(raw: str | None) -> list[str]:
    if raw and raw.strip():
        return shlex.split(raw.strip())
    return [str(REPO_ROOT / "scripts" / "v5" / "launch_kitchen_scene.sh")]


def _kill_scene_processes() -> None:
    patterns = [
        "ros2 launch kitchen_robot_description gazebo.launch.py",
        "ros_gz_sim create",
        "/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge",
        "tray_pose_extractor_node",
        "tray_pose_adapter_node",
        "object_id_publisher_node",
        "static_transform_publisher",
        "robot_state_publisher",
        "gz sim",
    ]
    for pat in patterns:
        subprocess.run(["pkill", "-f", pat], check=False, capture_output=True, text=True)
    time.sleep(2.0)


def _detect_tray_pose_mode() -> str:
    topics = _list_topics()
    if "/tray_tracking/pose_stream_raw" in topics:
        return "dedicated"
    if "/tray_tracking/pose_stream" in topics:
        return "legacy_degraded"
    return "unknown"


def _run_manual_path(timeout_sec: float) -> dict[str, Any]:
    rows = _probe_topics(timeout_sec)
    return {
        "status": _path_status(rows),
        "tray_pose_mode": _detect_tray_pose_mode(),
        "topics": _rows_json(rows),
    }


def _run_auto_path(timeout_sec: float, launch_cmd: list[str], kill_before_auto: bool) -> dict[str, Any]:
    launch_proc: subprocess.Popen[str] | None = None
    blocked_reason = None
    rows: list[TopicProbeResult] = []
    try:
        if kill_before_auto:
            _kill_scene_processes()
        launch_proc = subprocess.Popen(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(2.0)
        rows = _probe_topics(timeout_sec)
    except FileNotFoundError as exc:
        blocked_reason = str(exc)
    except OSError as exc:
        blocked_reason = str(exc)
    finally:
        if launch_proc is not None and launch_proc.poll() is None:
            launch_proc.terminate()
            try:
                launch_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                launch_proc.kill()
                launch_proc.wait(timeout=5.0)
    return {
        "status": _path_status(rows, blocked_reason=blocked_reason),
        "blocked_reason": blocked_reason,
        "tray_pose_mode": _detect_tray_pose_mode() if not blocked_reason else "unknown",
        "launch_command": launch_cmd,
        "topics": _rows_json(rows),
    }


def _topic_row_by_name(path_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in path_report.get("topics", []):
        topic = row.get("topic")
        if isinstance(topic, str):
            out[topic] = row
    return out


def _build_parity_report(manual: dict[str, Any], auto: dict[str, Any]) -> dict[str, Any]:
    if manual.get("status") == STATUS_BLOCKED or auto.get("status") == STATUS_BLOCKED:
        reason = "manual_or_auto_path_blocked"
        if manual.get("status") == STATUS_BLOCKED and manual.get("blocked_reason"):
            reason = f"manual_blocked:{manual['blocked_reason']}"
        elif auto.get("status") == STATUS_BLOCKED and auto.get("blocked_reason"):
            reason = f"auto_blocked:{auto['blocked_reason']}"
        return {"status": STATUS_BLOCKED, "reason": reason, "per_topic": {}}

    manual_rows = _topic_row_by_name(manual)
    auto_rows = _topic_row_by_name(auto)
    per_topic: dict[str, dict[str, Any]] = {}
    all_match = True
    for topic in REQUIRED_TOPICS:
        manual_sample = bool(manual_rows.get(topic, {}).get("sample"))
        auto_sample = bool(auto_rows.get(topic, {}).get("sample"))
        match = manual_sample == auto_sample
        all_match = all_match and match
        per_topic[topic] = {
            "manual_sample": manual_sample,
            "auto_sample": auto_sample,
            "match": match,
        }
    return {
        "status": STATUS_PASS if all_match else STATUS_FAIL,
        "reason": "all_topics_match" if all_match else "sample_mismatch_detected",
        "per_topic": per_topic,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP1.5 remediation B runtime parity checker")
    parser.add_argument(
        "--mode",
        choices=("manual", "auto", "both"),
        default="both",
        help="Which startup path(s) to check.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=25.0,
        help="Per-topic timeout (seconds) for list/sample checks.",
    )
    parser.add_argument(
        "--auto-launch-cmd",
        default=None,
        help="Command used by auto path (default: scripts/v5/launch_kitchen_scene.sh).",
    )
    parser.add_argument(
        "--no-kill-before-auto",
        action="store_true",
        help="Do not kill existing scene/bridge processes before auto path launch.",
    )
    parser.add_argument("--output", default=None, help="Write JSON report to file.")
    parser.add_argument("--no-pretty", action="store_true", help="Emit compact JSON output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    launch_cmd = _parse_launch_cmd(args.auto_launch_cmd)
    kill_before_auto = not bool(args.no_kill_before_auto)

    manual_report: dict[str, Any] = _blocked_path("not_requested")
    auto_report: dict[str, Any] = _blocked_path("not_requested")

    if args.mode in ("manual", "both"):
        manual_report = _run_manual_path(args.timeout_sec)
    if args.mode in ("auto", "both"):
        auto_report = _run_auto_path(args.timeout_sec, launch_cmd, kill_before_auto)

    parity = _build_parity_report(manual_report, auto_report) if args.mode == "both" else {"status": STATUS_BLOCKED, "reason": "mode_not_both", "per_topic": {}}
    run_statuses = [manual_report.get("status"), auto_report.get("status"), parity.get("status")]
    overall_ok = STATUS_FAIL not in run_statuses and STATUS_BLOCKED not in run_statuses
    report = {
        "tool": "wp1_5_runtime_parity_check",
        "timestamp_utc": _utc_now_iso(),
        "required_topics": REQUIRED_TOPICS,
        "config": {
            "mode": args.mode,
            "timeout_sec": args.timeout_sec,
            "auto_launch_command": launch_cmd,
            "kill_before_auto": kill_before_auto,
        },
        "paths": {
            "manual": manual_report,
            "auto": auto_report,
        },
        "parity": parity,
        "tray_pose_mode": auto_report.get("tray_pose_mode") if args.mode in ("auto", "both") else manual_report.get("tray_pose_mode"),
        "overall": {
            "result": STATUS_PASS if overall_ok else STATUS_FAIL,
            "pass": overall_ok,
        },
    }
    write_json(report, output_path=args.output, pretty=not args.no_pretty)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
