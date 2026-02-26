"""Common helpers for WP0 tools (config, json IO, optional ROS imports)."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "v5_wp0_diagnostics.yaml"


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    data.setdefault("_meta", {})["config_path"] = str(cfg_path)
    return data


def write_json(data: dict[str, Any], output_path: str | None = None, pretty: bool = True) -> None:
    payload = json.dumps(_sanitize_for_json(data), indent=2 if pretty else None, sort_keys=True)
    if output_path:
        Path(output_path).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help=f"YAML config path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--output", default=None, help="Write JSON output to file instead of stdout")
    parser.add_argument("--no-pretty", action="store_true", help="Emit compact JSON")


def maybe_load_ros() -> tuple[Any, Any]:
    try:
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
    except Exception as exc:
        raise RuntimeError("ROS 2 Python (rclpy) is required for live mode") from exc
    return rclpy, Node


def load_msg_class(type_name: str) -> type:
    """Load a ROS message class from `pkg/msg/Type` or `pkg.msg.Type`."""
    if "/msg/" in type_name:
        pkg, _, cls_name = type_name.partition("/msg/")
        module_name = f"{pkg}.msg"
    else:
        parts = type_name.split(".")
        if len(parts) < 3:
            raise ValueError(f"Unsupported type format: {type_name}")
        module_name = ".".join(parts[:-1])
        cls_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def stamp_to_ns(stamp: Any) -> int:
    sec = int(getattr(stamp, "sec"))
    nanosec = int(getattr(stamp, "nanosec"))
    return sec * 1_000_000_000 + nanosec


def get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    if not path:
        return cur
    tokens: list[str] = []
    i = 0
    while i < len(path):
        if path[i] == ".":
            i += 1
            continue
        if path[i] == "[":
            j = path.index("]", i)
            tokens.append(path[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < len(path) and path[j] not in ".[":
            j += 1
        tokens.append(path[i:j])
        i = j

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("[") and tok.endswith("]"):
            idx = int(tok[1:-1])
            cur = cur[idx]
            continue
        if isinstance(cur, dict):
            cur = cur[tok]
        else:
            cur = getattr(cur, tok)
    return cur


def now_wall_ns() -> int:
    return time.time_ns()


def run_subprocess(cmd: list[str], timeout_sec: float | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "success": completed.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"command": cmd, "error": str(exc), "success": False, "returncode": 127}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "error": f"timeout after {timeout_sec}s",
            "stdout": (exc.stdout or "")[-4000:],
            "stderr": (exc.stderr or "")[-4000:],
            "success": False,
            "returncode": None,
        }


def tool_result(tool: str, config: dict[str, Any], metrics: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
    result = {
        "tool": tool,
        "config_path": config.get("_meta", {}).get("config_path"),
        "metrics": metrics,
        "timestamp_unix_ns": now_wall_ns(),
    }
    if ok is not None:
        result["ok"] = bool(ok)
    return result


def finalize_output(data: dict[str, Any], args: argparse.Namespace) -> None:
    write_json(data, output_path=args.output, pretty=not args.no_pretty)
