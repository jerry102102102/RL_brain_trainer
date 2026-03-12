from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .common import load_yaml, write_json
except ImportError:  # pragma: no cover - script execution fallback
    from hrl_trainer.v5.tools.common import load_yaml, write_json


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"

THIS_FILE = Path(__file__).resolve()
PKG_ROOT = THIS_FILE.parents[3]
REPO_ROOT = THIS_FILE.parents[6]
DEFAULT_CONFIG_PATH = PKG_ROOT / "config" / "wp0_config.yaml"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "wp0"


@dataclass(frozen=True)
class ToolRunResult:
    command: list[str]
    cwd: str
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    json_output: dict[str, Any] | None
    error: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _run_capture(cmd: list[str], cwd: Path | None = None, timeout_sec: float | None = None) -> dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "success": cp.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc), "success": False, "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": (exc.stdout or ""),
            "stderr": (exc.stderr or ""),
            "success": False,
            "error": f"timeout after {timeout_sec}s",
        }


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = str(PKG_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = py_path if not existing else f"{py_path}:{existing}"
    return env


def _run_tool_module(
    module_name: str,
    *,
    config_path: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
    timeout_sec: float | None = None,
) -> ToolRunResult:
    cmd = [sys.executable, "-m", module_name, "--config", str(config_path), "--output", str(output_path)]
    if extra_args:
        cmd.extend(extra_args)
    try:
        if output_path.exists():
            output_path.unlink()
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=_py_env(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        parsed = None
        err = None
        if output_path.exists():
            try:
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive
                err = f"failed to parse tool JSON: {exc}"
        return ToolRunResult(
            command=cmd,
            cwd=str(cwd or Path.cwd()),
            returncode=cp.returncode,
            stdout_tail=cp.stdout[-4000:],
            stderr_tail=cp.stderr[-4000:],
            json_output=parsed,
            error=err,
        )
    except FileNotFoundError as exc:
        return ToolRunResult(cmd, str(cwd or Path.cwd()), 127, "", str(exc), None, error=str(exc))
    except subprocess.TimeoutExpired as exc:
        return ToolRunResult(
            cmd,
            str(cwd or Path.cwd()),
            None,
            (exc.stdout or "")[-4000:],
            (exc.stderr or "")[-4000:],
            None,
            error=f"timeout after {timeout_sec}s",
        )


def _section(status: str, *, summary: str, numeric_evidence: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None, subchecks: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "numeric_evidence": numeric_evidence or {},
        "evidence": evidence or {},
        "subchecks": subchecks or {},
    }


def _blocked(summary: str, reason: str, *, numeric_evidence: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None, subchecks: dict[str, Any] | None = None) -> dict[str, Any]:
    out = _section(
        STATUS_BLOCKED,
        summary=summary,
        numeric_evidence=numeric_evidence,
        evidence=evidence,
        subchecks=subchecks,
    )
    out["blocked_reason"] = reason
    return out


def _status_from_bool(ok: bool) -> str:
    return STATUS_PASS if ok else STATUS_FAIL


def _safe_get(d: dict[str, Any] | None, *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _replay_topic(topic: str) -> str:
    topic_s = str(topic).strip()
    if not topic_s.startswith("/"):
        topic_s = f"/{topic_s}"
    return f"/replay{topic_s}"


def _build_replay_topic_map(wp0_cfg: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    cameras = wp0_cfg.get("cameras", {})
    if not isinstance(cameras, dict):
        return mapping
    for cam in cameras.values():
        if not isinstance(cam, dict):
            continue
        for key in ("image_topic", "camera_info_topic"):
            topic = cam.get(key)
            if isinstance(topic, str) and topic.strip():
                mapping[topic] = _replay_topic(topic)
    return mapping


def collect_system_metadata(repo_root: Path) -> dict[str, Any]:
    git_hash = _run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
    uname = _run_capture(["uname", "-r"])
    nvidia = _run_capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    lsb = _run_capture(["bash", "-lc", "source /etc/os-release >/dev/null 2>&1; echo ${ID:-}:${VERSION_ID:-}:${PRETTY_NAME:-}"])

    wsl2 = False
    uname_r = (uname.get("stdout") or "").strip()
    if "WSL2" in uname_r.upper():
        wsl2 = True
    if os.environ.get("WSL_DISTRO_NAME"):
        wsl2 = True

    ubuntu_version = None
    pretty_os = None
    lsb_out = (lsb.get("stdout") or "").strip()
    if lsb_out:
        parts = lsb_out.split(":", 2)
        if len(parts) == 3:
            _, ubuntu_version, pretty_os = parts

    gpu_rows: list[dict[str, str]] = []
    for line in (nvidia.get("stdout") or "").splitlines():
        if not line.strip():
            continue
        cols = [x.strip() for x in line.split(",", 1)]
        if len(cols) == 2:
            gpu_rows.append({"gpu": cols[0], "driver": cols[1]})
        else:
            gpu_rows.append({"gpu": cols[0], "driver": ""})

    gpu_driver = None
    if gpu_rows:
        gpu_driver = {
            "gpus": gpu_rows,
            "driver_versions": sorted({r["driver"] for r in gpu_rows if r.get("driver")}),
        }

    ros_distro = os.environ.get("ROS_DISTRO")
    return {
        "timestamp_utc": _utc_now_iso(),
        "git_commit_hash": (git_hash.get("stdout") or "").strip() or None,
        "ros_distro": ros_distro,
        "ubuntu_version": ubuntu_version,
        "os_pretty_name": pretty_os,
        "wsl2": wsl2,
        "kernel_release": uname_r or None,
        "gpu_driver": gpu_driver,
    }


def build_report_skeleton(config_path: Path, artifacts_dir: Path, cfg: dict[str, Any], system_metadata: dict[str, Any]) -> dict[str, Any]:
    wp0 = cfg.get("wp0", {})
    frames_pdf = artifacts_dir / "frames.pdf"
    report_path = artifacts_dir / "wp0_report.json"
    return {
        "schema_version": "wp0_report.v1",
        "generated_at_utc": _utc_now_iso(),
        "system": system_metadata,
        "config": {
            "path": str(config_path),
            "use_sim_time": bool(wp0.get("use_sim_time", False)),
            "window_sec": float(wp0.get("window_sec", 0.0) or 0.0),
        },
        "artifacts": {
            "root": str(artifacts_dir),
            "wp0_report_json": str(report_path),
            "frames_pdf": str(frames_pdf),
        },
        "sections": {},
        "issues": [],
        "overall": {
            "result": STATUS_FAIL,
            "counts": {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_BLOCKED: 0},
        },
    }


def add_issue(report: dict[str, Any], section: str, status: str, reason: str, suggested_fix: str) -> None:
    report.setdefault("issues", []).append(
        {
            "section": section,
            "status": status,
            "reason": reason,
            "suggested_fix": suggested_fix,
        }
    )


def finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    counts = {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_BLOCKED: 0}
    for sec in report.get("sections", {}).values():
        st = sec.get("status")
        if st in counts:
            counts[st] += 1
    report["overall"]["counts"] = counts
    # Non-negotiable gate: any FAIL or BLOCKED prevents PASS.
    report["overall"]["result"] = STATUS_PASS if counts[STATUS_FAIL] == 0 and counts[STATUS_BLOCKED] == 0 else STATUS_FAIL
    report["overall"]["pass"] = report["overall"]["result"] == STATUS_PASS
    return report


def evaluate_camera_contract(
    cfg: dict[str, Any],
    live_image_tool: dict[str, Any] | None,
) -> dict[str, Any]:
    wp0 = cfg["wp0"]
    cams = wp0.get("cameras", {})
    per_camera: dict[str, Any] = {}
    numeric: dict[str, Any] = {"camera_count": len(cams)}
    subchecks: dict[str, Any] = {}

    live_per_topic = _safe_get(live_image_tool, "metrics", "per_topic") if live_image_tool else None
    live_available = isinstance(live_per_topic, dict)
    config_ok = True
    runtime_ok = True
    runtime_blocked_reasons: list[str] = []

    for cam_name, cam in cams.items():
        topic = cam.get("image_topic")
        exp = {
            "topic": topic,
            "type": cam.get("image_type"),
            "encoding": cam.get("encoding"),
            "width": cam.get("width"),
            "height": cam.get("height"),
            "fps": cam.get("expected_fps"),
        }
        cam_entry: dict[str, Any] = {"expected": exp}

        for req_key in ("topic", "type", "encoding", "width", "height", "fps"):
            if exp.get(req_key) in (None, "", 0):
                config_ok = False
                cam_entry.setdefault("config_errors", []).append(f"missing_expected_{req_key}")

        obs = None
        if live_available and topic in live_per_topic:
            live_m = live_per_topic[topic]
            obs_contract = live_m.get("observed_contract") or {}
            obs = {
                "type": obs_contract.get("message_type"),
                "encoding": obs_contract.get("encoding"),
                "width": obs_contract.get("width"),
                "height": obs_contract.get("height"),
                "fps": live_m.get("fps"),
                "latency_p95_ms": _safe_get(live_m, "latency", "p95_ms"),
            }
            mismatches: list[str] = []
            if exp.get("type") and obs.get("type") and exp["type"] != obs["type"]:
                mismatches.append(f"type expected={exp['type']} observed={obs['type']}")
            if exp.get("encoding") and obs.get("encoding") and exp["encoding"] != obs["encoding"]:
                mismatches.append(f"encoding expected={exp['encoding']} observed={obs['encoding']}")
            if exp.get("width") is not None and obs.get("width") is not None and int(exp["width"]) != int(obs["width"]):
                mismatches.append(f"width expected={exp['width']} observed={obs['width']}")
            if exp.get("height") is not None and obs.get("height") is not None and int(exp["height"]) != int(obs["height"]):
                mismatches.append(f"height expected={exp['height']} observed={obs['height']}")
            fps = obs.get("fps")
            if fps is None:
                mismatches.append("observed_fps_missing")
            elif float(fps) <= 0:
                mismatches.append("observed_fps_nonpositive")
            lat_gate = _safe_get(live_m, "latency", "gate", "pass")
            if lat_gate is False:
                mismatches.append("image_latency_gate_failed")
            cam_entry["observed"] = obs
            if mismatches:
                runtime_ok = False
                cam_entry["mismatches"] = mismatches
        else:
            runtime_ok = False
            runtime_blocked_reasons.append(f"no_live_metrics_for_{cam_name}")

        per_camera[cam_name] = cam_entry
        if obs:
            numeric[f"{cam_name}_fps"] = obs.get("fps")
            numeric[f"{cam_name}_width"] = obs.get("width")
            numeric[f"{cam_name}_height"] = obs.get("height")
            numeric[f"{cam_name}_latency_p95_ms"] = obs.get("latency_p95_ms")
        else:
            numeric[f"{cam_name}_fps"] = None

    subchecks["config_contract"] = {"status": _status_from_bool(config_ok)}
    if not live_image_tool:
        subchecks["live_contract_and_fps"] = {"status": STATUS_BLOCKED, "reason": "image_health_diag not run"}
    elif not live_available:
        subchecks["live_contract_and_fps"] = {"status": STATUS_BLOCKED, "reason": "image_health_diag produced no per_topic metrics"}
    elif runtime_blocked_reasons and not any(v.get("observed") for v in per_camera.values()):
        subchecks["live_contract_and_fps"] = {"status": STATUS_BLOCKED, "reason": "; ".join(runtime_blocked_reasons)}
    else:
        subchecks["live_contract_and_fps"] = {"status": _status_from_bool(runtime_ok)}

    if subchecks["config_contract"]["status"] == STATUS_FAIL:
        return _section(
            STATUS_FAIL,
            summary="Camera contract config is incomplete or invalid.",
            numeric_evidence=numeric,
            evidence={"per_camera": per_camera},
            subchecks=subchecks,
        )
    live_status = subchecks["live_contract_and_fps"]["status"]
    if live_status == STATUS_BLOCKED:
        return _blocked(
            "Camera contract config is present, but live camera contract/fps validation is unavailable.",
            subchecks["live_contract_and_fps"].get("reason", "live image metrics unavailable"),
            numeric_evidence=numeric,
            evidence={"per_camera": per_camera},
            subchecks=subchecks,
        )
    return _section(
        STATUS_PASS if runtime_ok else STATUS_FAIL,
        summary="Camera contract topic/type/encoding/resolution/fps validated." if runtime_ok else "Camera contract live validation failed.",
        numeric_evidence=numeric,
        evidence={"per_camera": per_camera},
        subchecks=subchecks,
    )


def evaluate_tf_contract(cfg: dict[str, Any], tf_tool: dict[str, Any] | None, frames_pdf: Path) -> dict[str, Any]:
    required_pairs = cfg["wp0"].get("tf_checks", {}).get("required_pairs", [])
    frames_pdf_exists = frames_pdf.exists()
    numeric = {"required_pairs": len(required_pairs), "tf_echo_successes": 0, "view_frames_success": 0}
    evidence = {"frames_pdf_path": str(frames_pdf), "frames_pdf_exists": frames_pdf_exists, "required_pairs": required_pairs}
    subchecks: dict[str, Any] = {
        "tf_contract_pairs_present": {"status": _status_from_bool(len(required_pairs) > 0)},
        "frames_pdf_under_artifacts_wp0": {"status": _status_from_bool(str(frames_pdf).endswith("/artifacts/wp0/frames.pdf"))},
    }
    if not tf_tool:
        subchecks["tf_runtime"] = {"status": STATUS_BLOCKED, "reason": "tf_check_helper not run"}
        return _blocked("TF contract config present; runtime TF validation unavailable.", "tf_check_helper not run", numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)

    checks = _safe_get(tf_tool, "metrics", "checks") or []
    runner = _safe_get(tf_tool, "_runner") if isinstance(tf_tool, dict) else None
    returncode = runner.get("returncode") if isinstance(runner, dict) else None
    stderr_tail = (runner.get("stderr_tail") if isinstance(runner, dict) else "") or ""
    evidence["tool"] = {"ok": tf_tool.get("ok"), "returncode": returncode}
    evidence["checks"] = checks
    if checks:
        numeric["view_frames_success"] = 1 if bool(checks[0].get("success")) else 0
        numeric["tf_echo_successes"] = int(sum(1 for row in checks[1:] if row.get("success")))
    runtime_ok = bool(checks) and all(bool(row.get("success")) for row in checks) and frames_pdf_exists
    subchecks["tf_runtime"] = {"status": _status_from_bool(runtime_ok) if checks else STATUS_BLOCKED}
    if checks:
        subchecks["frames_pdf_generated"] = {"status": _status_from_bool(frames_pdf_exists)}
    else:
        subchecks["frames_pdf_generated"] = {"status": STATUS_BLOCKED, "reason": "view_frames not executed"}
    if not checks:
        stderr_compact = " ".join(stderr_tail.strip().split())
        stderr_summary = stderr_compact[-240:] if stderr_compact else "<empty>"
        missing_reason = f"tf_check_helper produced no metrics.checks (returncode={returncode}, stderr_tail={stderr_summary})"
        subchecks["tf_runtime"]["reason"] = missing_reason
        return _blocked(
            "TF runtime checks unavailable.",
            missing_reason,
            numeric_evidence=numeric,
            evidence=evidence,
            subchecks=subchecks,
        )
    return _section(
        STATUS_PASS if runtime_ok else STATUS_FAIL,
        summary="TF contract and frames.pdf output validated." if runtime_ok else "TF contract runtime check failed or frames.pdf missing.",
        numeric_evidence=numeric,
        evidence=evidence,
        subchecks=subchecks,
    )


def evaluate_approx_sync(cfg: dict[str, Any], approx_tool: dict[str, Any] | None) -> dict[str, Any]:
    wp0 = cfg["wp0"]
    threshold_slop = float(wp0["thresholds"]["approx_sync_slop_ms"])
    threshold_success = float(wp0["thresholds"]["approx_sync_success_rate_min"])
    queue_size = int(wp0.get("approx_sync", {}).get("queue_size", 10))
    numeric = {
        "configured_slop_ms": threshold_slop,
        "configured_queue_size": queue_size,
        "success_rate_threshold": threshold_success,
        "success_rate": None,
        "pairs": None,
    }
    subchecks = {
        "slop_ms_equals_50": {"status": _status_from_bool(abs(threshold_slop - 50.0) < 1e-9)},
        "queue_size_default_10": {"status": _status_from_bool(queue_size == 10)},
    }
    if not approx_tool or not isinstance(_safe_get(approx_tool, "metrics"), dict):
        subchecks["runtime_success_rate"] = {"status": STATUS_BLOCKED, "reason": "approx_sync_eval not run"}
        return _blocked(
            "Approx sync config locked, but runtime success-rate validation unavailable.",
            "approx_sync_eval not run",
            numeric_evidence=numeric,
            evidence={},
            subchecks=subchecks,
        )

    metrics = approx_tool["metrics"]
    success_rate = metrics.get("success_rate")
    pairs = metrics.get("pairs")
    slop_ms_observed = metrics.get("slop_ms")
    queue_observed = metrics.get("queue_size")
    numeric.update(
        {
            "success_rate": success_rate,
            "pairs": pairs,
            "observed_slop_ms": slop_ms_observed,
            "observed_queue_size": queue_observed,
        }
    )
    runtime_ok = (
        success_rate is not None
        and float(success_rate) > threshold_success
        and slop_ms_observed is not None
        and abs(float(slop_ms_observed) - 50.0) < 1e-9
        and queue_observed == 10
    )
    subchecks["runtime_success_rate"] = {"status": _status_from_bool(runtime_ok)}
    return _section(
        STATUS_PASS if runtime_ok and subchecks["slop_ms_equals_50"]["status"] == STATUS_PASS and subchecks["queue_size_default_10"]["status"] == STATUS_PASS else STATUS_FAIL,
        summary="Approx sync slop/queue size/success rate validated." if runtime_ok else "Approx sync requirement failed.",
        numeric_evidence=numeric,
        evidence={"metrics": metrics},
        subchecks=subchecks,
    )


def evaluate_tray_stability(cfg: dict[str, Any], pose_tool: dict[str, Any] | None, id_tool: dict[str, Any] | None) -> dict[str, Any]:
    th = cfg["wp0"]["thresholds"]
    std_limit_m = float(th["pose_jitter_std_m"])
    id_switch_limit = float(th["id_switch_rate_max"])
    numeric = {
        "jitter_std_limit_m": std_limit_m,
        "id_switch_rate_limit": id_switch_limit,
        "jitter_std_x_m": None,
        "jitter_std_y_m": None,
        "jitter_std_z_m": None,
        "id_switch_rate": None,
        "missing_rate": None,
    }
    subchecks: dict[str, Any] = {}
    evidence: dict[str, Any] = {}

    pose_metrics = _safe_get(pose_tool, "metrics") if pose_tool else None
    id_metrics = _safe_get(id_tool, "metrics") if id_tool else None
    id_status = id_tool.get("status") if isinstance(id_tool, dict) else None
    id_blocked_reason = _safe_get(id_tool, "blocked_reason") if isinstance(id_tool, dict) else None

    if isinstance(pose_metrics, dict):
        std_xyz = pose_metrics.get("std_xyz_m") or []
        if len(std_xyz) == 3:
            numeric["jitter_std_x_m"], numeric["jitter_std_y_m"], numeric["jitter_std_z_m"] = std_xyz
        subchecks["tray_jitter_xyz_std_lt_3mm"] = {
            "status": _status_from_bool(bool(_safe_get(pose_metrics, "gate", "pass"))),
        }
        evidence["pose_jitter"] = pose_metrics
    else:
        subchecks["tray_jitter_xyz_std_lt_3mm"] = {"status": STATUS_BLOCKED, "reason": "pose_jitter_eval not run"}

    if id_status == STATUS_BLOCKED:
        reason = str(id_blocked_reason or "id_switch_eval blocked")
        subchecks["id_switch_lt_1pct"] = {"status": STATUS_BLOCKED, "reason": reason}
        subchecks["missing_rate_reported"] = {"status": STATUS_BLOCKED, "reason": reason}
        evidence["id_switch"] = id_metrics if isinstance(id_metrics, dict) else {"blocked_reason": reason}
    elif isinstance(id_metrics, dict):
        numeric["id_switch_rate"] = id_metrics.get("switch_rate")
        numeric["missing_rate"] = id_metrics.get("missing_rate")
        switch_rate = id_metrics.get("switch_rate")
        id_ok = switch_rate is not None and float(switch_rate) < id_switch_limit
        subchecks["id_switch_lt_1pct"] = {"status": _status_from_bool(id_ok)}
        subchecks["missing_rate_reported"] = {"status": _status_from_bool(id_metrics.get("missing_rate") is not None)}
        evidence["id_switch"] = id_metrics
    else:
        subchecks["id_switch_lt_1pct"] = {"status": STATUS_BLOCKED, "reason": "id_switch_eval not run"}
        subchecks["missing_rate_reported"] = {"status": STATUS_BLOCKED, "reason": "id_switch_eval not run"}

    statuses = [v.get("status") for v in subchecks.values()]
    if STATUS_FAIL in statuses:
        status = STATUS_FAIL
    elif all(s == STATUS_PASS for s in statuses):
        status = STATUS_PASS
    else:
        status = STATUS_BLOCKED
    summary = {
        STATUS_PASS: "Tray jitter, id-switch, and missing-rate requirements validated.",
        STATUS_FAIL: "Tray stability requirement failed.",
        STATUS_BLOCKED: "Tray stability validation incomplete (missing live/jsonl inputs).",
    }[status]
    if status == STATUS_BLOCKED:
        out = _blocked(summary, "pose_jitter_eval and/or id_switch_eval unavailable", numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)
        return out
    return _section(status, summary=summary, numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)


def evaluate_state_latency(cfg: dict[str, Any], state_tool: dict[str, Any] | None) -> dict[str, Any]:
    limit_ms = float(cfg["wp0"]["thresholds"]["state_latency_p95_ms"])
    numeric = {
        "p95_limit_ms": limit_ms,
        "p95_ms": None,
        "sample_count": None,
    }
    subchecks = {
        "p95_lt_80ms": {"status": STATUS_BLOCKED, "reason": "state_latency_eval not run"},
        "same_clock_domain_definition": {"status": STATUS_BLOCKED, "reason": "state_latency_eval not run"},
    }
    evidence: dict[str, Any] = {}
    if not state_tool or not isinstance(_safe_get(state_tool, "metrics"), dict):
        return _blocked(
            "State topic latency validation unavailable.",
            "state_latency_eval not run",
            numeric_evidence=numeric,
            evidence=evidence,
            subchecks=subchecks,
        )

    metrics = state_tool["metrics"]
    overall = metrics.get("overall", {})
    numeric["p95_ms"] = _safe_get(overall, "p95_ms")
    numeric["sample_count"] = overall.get("count")
    p95_ok = bool(_safe_get(overall, "gate", "pass"))
    definition = metrics.get("latency_definition")
    same_clock_ok = isinstance(definition, str) and "same ROS time basis" in definition
    subchecks["p95_lt_80ms"] = {"status": _status_from_bool(p95_ok)}
    subchecks["same_clock_domain_definition"] = {"status": _status_from_bool(same_clock_ok)}
    evidence["metrics"] = metrics
    status = STATUS_PASS if p95_ok and same_clock_ok else STATUS_FAIL
    return _section(
        status,
        summary="State topic latency requirement validated." if status == STATUS_PASS else "State topic latency requirement failed.",
        numeric_evidence=numeric,
        evidence=evidence,
        subchecks=subchecks,
    )


def evaluate_rosbag_replay(
    cfg: dict[str, Any],
    rosbag_print_tool: dict[str, Any] | None,
    replay_image_tool: dict[str, Any] | None,
    replay_bag_path: str | None,
) -> dict[str, Any]:
    wp0 = cfg["wp0"]
    replay_limit = float(wp0["thresholds"]["replay_image_latency_p95_ms"])
    default_bag = _safe_get(wp0, "rosbag", "default_bag_path")
    numeric = {
        "replay_image_latency_p95_limit_ms": replay_limit,
        "record_topics_count": 0,
        "replay_image_latency_p95_ms": None,
    }
    subchecks: dict[str, Any] = {}
    evidence: dict[str, Any] = {"bag_path": replay_bag_path or default_bag}

    if rosbag_print_tool and isinstance(_safe_get(rosbag_print_tool, "metrics"), dict):
        metrics = rosbag_print_tool["metrics"]
        rec_cmd = _safe_get(metrics, "record", "command") or []
        rep_cmd = _safe_get(metrics, "replay", "command") or []
        evidence["commands"] = {
            "record": _safe_get(metrics, "record", "shell"),
            "replay": _safe_get(metrics, "replay", "shell"),
        }
        numeric["record_topics_count"] = max(0, len(rec_cmd) - 6) if isinstance(rec_cmd, list) else 0
        record_has_sim_time = "--use-sim-time" in rec_cmd if isinstance(rec_cmd, list) else False
        replay_has_clock = "--clock" in rep_cmd if isinstance(rep_cmd, list) else False
        subchecks["record_command_uses_sim_time"] = {"status": _status_from_bool(record_has_sim_time)}
        subchecks["replay_command_uses_clock"] = {"status": _status_from_bool(replay_has_clock)}
    else:
        subchecks["record_command_uses_sim_time"] = {"status": STATUS_BLOCKED, "reason": "rosbag2_helper print-commands not run"}
        subchecks["replay_command_uses_clock"] = {"status": STATUS_BLOCKED, "reason": "rosbag2_helper print-commands not run"}

    if replay_image_tool and isinstance(_safe_get(replay_image_tool, "metrics"), dict):
        per_topic = _safe_get(replay_image_tool, "metrics", "per_topic") or {}
        replay_topic_map = _safe_get(replay_image_tool, "_replay", "replay_topic_map")
        remapped_topics = set()
        if isinstance(replay_topic_map, dict):
            remapped_topics = {str(v) for v in replay_topic_map.values() if str(v).startswith("/replay/v5/")}
            evidence["replay_topic_map"] = replay_topic_map
            evidence["replay_topic_filter"] = sorted(remapped_topics)
        p95_vals = []
        ignored_topics: list[str] = []
        included_topics = 0
        for topic, topic_metrics in per_topic.items():
            if remapped_topics and topic not in remapped_topics:
                ignored_topics.append(topic)
                continue
            p95 = _safe_get(topic_metrics, "latency", "p95_ms")
            if p95 is not None:
                p95_vals.append(float(p95))
            included_topics += 1
            evidence.setdefault("replay_image_topics", {})[topic] = {
                "latency_p95_ms": p95,
                "frames": topic_metrics.get("frames"),
            }
        if ignored_topics:
            evidence["ignored_non_remapped_replay_topics"] = sorted(ignored_topics)
        if remapped_topics and included_topics == 0:
            subchecks["replay_image_latency_p95_lt_120ms"] = {
                "status": STATUS_BLOCKED,
                "reason": "no remapped /replay/v5/* replay image metrics",
            }
            statuses = [v.get("status") for v in subchecks.values()]
            if STATUS_FAIL in statuses:
                status = STATUS_FAIL
            elif all(s == STATUS_PASS for s in statuses):
                status = STATUS_PASS
            else:
                status = STATUS_BLOCKED
            summary = {
                STATUS_PASS: "Rosbag record/replay commands and replay image latency validated.",
                STATUS_FAIL: "Rosbag record/replay requirement failed.",
                STATUS_BLOCKED: "Rosbag record/replay validation incomplete (replay not executed or samples unavailable).",
            }[status]
            return _blocked(summary, "replay latency metrics unavailable", numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)
        if p95_vals:
            numeric["replay_image_latency_p95_ms"] = max(p95_vals)
            subchecks["replay_image_latency_p95_lt_120ms"] = {"status": _status_from_bool(max(p95_vals) < replay_limit)}
        else:
            subchecks["replay_image_latency_p95_lt_120ms"] = {"status": STATUS_BLOCKED, "reason": "no replay image latency samples"}
    else:
        reason = "replay bag not provided/run" if not replay_bag_path else "replay image diagnostic unavailable"
        subchecks["replay_image_latency_p95_lt_120ms"] = {"status": STATUS_BLOCKED, "reason": reason}

    statuses = [v.get("status") for v in subchecks.values()]
    if STATUS_FAIL in statuses:
        status = STATUS_FAIL
    elif all(s == STATUS_PASS for s in statuses):
        status = STATUS_PASS
    else:
        status = STATUS_BLOCKED

    summary = {
        STATUS_PASS: "Rosbag record/replay commands and replay image latency validated.",
        STATUS_FAIL: "Rosbag record/replay requirement failed.",
        STATUS_BLOCKED: "Rosbag record/replay validation incomplete (replay not executed or samples unavailable).",
    }[status]
    if status == STATUS_BLOCKED:
        return _blocked(summary, "replay latency metrics unavailable", numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)
    return _section(status, summary=summary, numeric_evidence=numeric, evidence=evidence, subchecks=subchecks)


def attach_runner_metadata(tool_out: dict[str, Any] | None, run_res: ToolRunResult | None) -> dict[str, Any] | None:
    if run_res is None:
        return tool_out
    out = dict(tool_out) if isinstance(tool_out, dict) else {}
    out["_runner"] = {
        "command": run_res.command,
        "cwd": run_res.cwd,
        "returncode": run_res.returncode,
        "stdout_tail": run_res.stdout_tail,
        "stderr_tail": run_res.stderr_tail,
        "error": run_res.error,
    }
    return out


def _add_section_issues(report: dict[str, Any], name: str, section: dict[str, Any]) -> None:
    status = section.get("status")
    subchecks = section.get("subchecks", {})
    if status == STATUS_FAIL:
        add_issue(report, name, STATUS_FAIL, section.get("summary", "section failed"), "Inspect `sections.%s.evidence` and rerun after fixing the failing contract/metric." % name)
    if status == STATUS_BLOCKED:
        add_issue(
            report,
            name,
            STATUS_BLOCKED,
            section.get("blocked_reason", section.get("summary", "section blocked")),
            "Provide the required runtime inputs (ROS graph, topics, or bag path) and rerun wp0_healthcheck.",
        )
    for sub_name, sub in subchecks.items():
        if sub.get("status") == STATUS_FAIL:
            add_issue(
                report,
                name,
                STATUS_FAIL,
                f"{sub_name} failed",
                "Fix the threshold/config mismatch or runtime behavior, then rerun the healthcheck.",
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WP0 DoD healthcheck runner and unified report generator.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="WP0 healthcheck YAML config path")
    p.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Artifacts output directory (frames.pdf, per-tool JSON, wp0_report.json)")
    p.add_argument("--output", default=None, help="Explicit path for wp0_report.json (default: <artifacts-dir>/wp0_report.json)")
    p.add_argument("--live", action="store_true", help="Run live ROS-backed checks (TF/image/approx sync/state/pose; id-switch depends on config mode)")
    p.add_argument("--duration-sec", type=float, default=None, help="Override wp0.window_sec for live/replay diagnostics")
    p.add_argument("--replay-bag", default=None, help="Optional rosbag path to run replay image latency check (requires ros2 bag play)")
    p.add_argument("--tf-timeout-sec", type=float, default=5.0, help="Timeout for TF helper subcommands")
    p.add_argument("--no-pretty", action="store_true", help="Write compact JSON")
    return p


def _run_rosbag_replay_image_diag(config_path: Path, artifacts_dir: Path, duration_sec: float, bag_path: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    out_path = artifacts_dir / "replay_image_health.json"
    cfg = load_yaml(config_path)
    replay_topic_map = _build_replay_topic_map(cfg.get("wp0", {}))
    cmd = ["ros2", "bag", "play", bag_path, "--clock"]
    if replay_topic_map:
        cmd.extend(["--remap", *[f"{src}:={dst}" for src, dst in replay_topic_map.items()]])
    proc = None
    meta: dict[str, Any] = {"bag_play_command": cmd, "bag_path": bag_path, "replay_topic_map": replay_topic_map}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(artifacts_dir),
            env=_py_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        meta["error"] = str(exc)
        return None, meta
    except Exception as exc:  # pragma: no cover - defensive
        meta["error"] = str(exc)
        return None, meta

    try:
        run_res = _run_tool_module(
            "hrl_trainer.v5.tools.image_health_diag",
            config_path=config_path,
            output_path=out_path,
            extra_args=["--duration-sec", str(duration_sec), "--replay-mode"],
            cwd=REPO_ROOT,
            timeout_sec=max(20.0, duration_sec + 10.0),
        )
        tool_json = attach_runner_metadata(run_res.json_output, run_res)
        meta["diag_returncode"] = run_res.returncode
        return tool_json, meta
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else artifacts_dir / "wp0_report.json"

    cfg = load_yaml(config_path)
    wp0 = cfg.get("wp0", {})
    duration_sec = float(args.duration_sec or wp0.get("window_sec", 60.0))
    system_meta = collect_system_metadata(REPO_ROOT)
    report = build_report_skeleton(config_path, artifacts_dir, cfg, system_meta)

    tool_outputs: dict[str, dict[str, Any] | None] = {}

    # Always capture rosbag helper command contract (no ROS runtime needed).
    rosbag_print_run = _run_tool_module(
        "hrl_trainer.v5.tools.rosbag2_helper",
        config_path=config_path,
        output_path=artifacts_dir / "rosbag_print_commands.json",
        extra_args=["print-commands"],
        cwd=REPO_ROOT,
        timeout_sec=10.0,
    )
    tool_outputs["rosbag_print"] = attach_runner_metadata(rosbag_print_run.json_output, rosbag_print_run)

    if args.live:
        tf_pairs = wp0.get("tf_checks", {}).get("required_pairs", [])
        tf_timeout_budget = max(
            15.0,
            5.0 + (len(tf_pairs) + 1) * float(args.tf_timeout_sec) + 5.0,
        )
        tf_run = _run_tool_module(
            "hrl_trainer.v5.tools.tf_check_helper",
            config_path=config_path,
            output_path=artifacts_dir / "tf_check.json",
            extra_args=["--timeout-sec", str(args.tf_timeout_sec)],
            cwd=artifacts_dir,
            timeout_sec=tf_timeout_budget,
        )
        tool_outputs["tf"] = attach_runner_metadata(tf_run.json_output, tf_run)

        for key, module in (
            ("image_live", "hrl_trainer.v5.tools.image_health_diag"),
            ("approx_sync", "hrl_trainer.v5.tools.approx_sync_eval"),
            ("pose_jitter", "hrl_trainer.v5.tools.pose_jitter_eval"),
            ("state_latency", "hrl_trainer.v5.tools.state_latency_eval"),
        ):
            run = _run_tool_module(
                module,
                config_path=config_path,
                output_path=artifacts_dir / f"{key}.json",
                extra_args=["--duration-sec", str(duration_sec)],
                cwd=REPO_ROOT,
                timeout_sec=max(20.0, duration_sec + 10.0),
            )
            tool_outputs[key] = attach_runner_metadata(run.json_output, run)

        id_run = _run_tool_module(
            "hrl_trainer.v5.tools.id_switch_eval",
            config_path=config_path,
            output_path=artifacts_dir / "id_switch.json",
            extra_args=["--duration-sec", str(duration_sec)],
            cwd=REPO_ROOT,
            timeout_sec=max(20.0, duration_sec + 10.0),
        )
        tool_outputs["id_switch"] = attach_runner_metadata(id_run.json_output, id_run)
    else:
        tool_outputs["tf"] = None
        tool_outputs["image_live"] = None
        tool_outputs["approx_sync"] = None
        tool_outputs["pose_jitter"] = None
        tool_outputs["state_latency"] = None
        tool_outputs["id_switch"] = None

    replay_tool = None
    replay_meta = None
    if args.replay_bag:
        replay_tool, replay_meta = _run_rosbag_replay_image_diag(config_path, artifacts_dir, duration_sec, args.replay_bag)
        if replay_tool is not None:
            replay_tool = dict(replay_tool)
            replay_tool["_replay"] = _json_safe(replay_meta)
        else:
            replay_tool = {"_replay": _json_safe(replay_meta)}
    tool_outputs["replay_image"] = replay_tool

    report["tool_outputs"] = tool_outputs

    frames_pdf = artifacts_dir / "frames.pdf"
    report["sections"]["camera_contract"] = evaluate_camera_contract(cfg, tool_outputs.get("image_live"))
    report["sections"]["tf_contract"] = evaluate_tf_contract(cfg, tool_outputs.get("tf"), frames_pdf)
    report["sections"]["approx_sync"] = evaluate_approx_sync(cfg, tool_outputs.get("approx_sync"))
    report["sections"]["tray_stability"] = evaluate_tray_stability(cfg, tool_outputs.get("pose_jitter"), tool_outputs.get("id_switch"))
    report["sections"]["state_latency"] = evaluate_state_latency(cfg, tool_outputs.get("state_latency"))
    report["sections"]["rosbag_replay"] = evaluate_rosbag_replay(
        cfg,
        tool_outputs.get("rosbag_print"),
        tool_outputs.get("replay_image"),
        args.replay_bag,
    )

    for sec_name, sec in report["sections"].items():
        _add_section_issues(report, sec_name, sec)
    finalize_report(report)
    write_json(report, output_path=str(output_path), pretty=not args.no_pretty)
    return 0 if report["overall"]["result"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
