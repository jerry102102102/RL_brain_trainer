"""End-to-end L1 client that turns a user task into RL-ready input.

The client orchestrates the Qwen-facing MCP bridge:

1. read L1 scene context,
2. ask an LLM/VLM backend for a structured tool-call proposal,
3. resolve the proposal into a validated IntentPacket,
4. prepare a dry-run Approach->Finisher skill request for the RL stack.

The default backend is deterministic (`mock_qwen`) so the integration can be
tested without loading a large model.  The same prompt and parser are used by
the optional `qwen_subprocess` backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from .qwen_mcp_tools import QwenMcpBridge


DEFAULT_QWEN_PYTHON = "/home/jerry/venvs/qwenvl/bin/python"
DEFAULT_QWEN_SCRIPT = str(Path(__file__).resolve().parent / "tools" / "qwenvl_text_runner.py")


@dataclass(frozen=True)
class L1ClientResult:
    backend: str
    user_command: str
    prompt: str
    model_text: str
    tool_call: dict[str, Any]
    intent_resolution: dict[str, Any]
    skill_request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "user_command": self.user_command,
            "prompt": self.prompt,
            "model_text": self.model_text,
            "tool_call": self.tool_call,
            "intent_resolution": self.intent_resolution,
            "skill_request": self.skill_request,
        }


def _compact_scene_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "known_objects": context.get("known_objects", []),
        "slots": [
            {
                "slot_id": slot["slot_id"],
                "allowed_objects": slot.get("allowed_objects", []),
                "center_xyz": slot.get("center_xyz"),
            }
            for slot in context.get("slots", [])
        ],
        "available_high_level_pipeline": context.get("available_high_level_pipeline", {}),
        "forbidden_control_outputs": context.get("forbidden_control_outputs", []),
    }


def build_l1_prompt(user_command: str, scene_context: Mapping[str, Any]) -> str:
    compact_context = _compact_scene_context(scene_context)
    return (
        "You are the L1 semantic task interpreter for a modular robot arm system.\n"
        "Your job is to select an object, source slot, target slot, constraints, and semantic subtasks.\n"
        "Do not output joint actions, trajectories, torques, delta_q, or raw controls.\n"
        "Semantic subtasks are allowed, but they must be high-level names/descriptions only.\n"
        "Return exactly one JSON object with this schema:\n"
        "{\"tool\":\"resolve_intent_packet\",\"arguments\":{\"object_id\":\"...\","
        "\"source_slot\":\"...\",\"target_slot\":\"...\",\"constraints\":{\"speed_cap\":\"SLOW\"},"
        "\"semantic_subtasks\":[{\"name\":\"pre_grasp_align\",\"description\":\"...\","
        "\"posture_constraint\":\"keep tray level\"}]}}\n\n"
        "For a tray move, use this semantic subtask sequence unless the scene context makes it invalid:\n"
        "pre_grasp_align -> under_tray_insert_pose -> level_lift -> carry_midline -> "
        "pre_insert_align -> stable_insert_hold.\n\n"
        f"User command: {user_command}\n\n"
        "Scene context JSON:\n"
        f"{json.dumps(compact_context, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object")
    return payload


def _slot_ids(scene_context: Mapping[str, Any]) -> list[str]:
    return [str(slot["slot_id"]) for slot in scene_context.get("slots", [])]


def mock_qwen_decision(user_command: str, scene_context: Mapping[str, Any]) -> str:
    """Deterministic Qwen-like decision for repeatable integration tests."""
    slots = _slot_ids(scene_context)
    if len(slots) < 2:
        raise ValueError("Need at least two slots for a move task")

    lower = user_command.lower()
    source = next((slot for slot in slots if slot.lower() in lower), slots[0])
    target = next((slot for slot in slots if slot.lower() in lower and slot != source), None)
    if target is None:
        for slot in slots:
            if slot != source:
                target = slot
                break
    object_id = "tray1"
    known_objects = scene_context.get("known_objects") or []
    if known_objects:
        object_id = str(known_objects[0])

    return json.dumps(
        {
            "tool": "resolve_intent_packet",
            "arguments": {
                "object_id": object_id,
                "source_slot": source,
                "target_slot": target,
                "constraints": {
                    "speed_cap": "SLOW",
                    "clearance_m": 0.02,
                    "timeout_s": 10.0,
                },
                "semantic_subtasks": [
                    {
                        "name": "pre_grasp_align",
                        "description": "Approach the source side with the EE already horizontal to the ground.",
                        "posture_constraint": "keep EE tray plane horizontal",
                    },
                    {
                        "name": "under_tray_insert_pose",
                        "description": "Slide forward under the virtual tray while preserving a horizontal EE.",
                        "posture_constraint": "keep EE tray plane horizontal",
                    },
                    {
                        "name": "level_lift",
                        "description": "Lift the tray without changing the horizontal EE posture.",
                        "posture_constraint": "keep EE tray plane horizontal",
                    },
                    {
                        "name": "carry_midline",
                        "description": "Carry across the local workspace with the tray plane held level.",
                        "posture_constraint": "keep EE tray plane horizontal",
                    },
                    {
                        "name": "pre_insert_align",
                        "description": "Align with the destination side while preserving the horizontal EE posture.",
                        "posture_constraint": "keep EE tray plane horizontal",
                    },
                    {
                        "name": "stable_insert_hold",
                        "description": "Settle and hold the destination insertion pose level with low motion.",
                        "posture_constraint": "keep EE tray plane horizontal and low motion",
                    },
                ],
            },
        },
        indent=2,
        sort_keys=True,
    )


def call_qwen_subprocess(
    prompt: str,
    *,
    qwen_python: str = DEFAULT_QWEN_PYTHON,
    qwen_script: str = DEFAULT_QWEN_SCRIPT,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    timeout_s: float = 180.0,
) -> str:
    script = Path(qwen_script)
    if not script.exists():
        raise FileNotFoundError(
            f"Qwen text runner not found: {script}. Use --backend mock_qwen or create the text runner."
        )
    proc = subprocess.run(
        [
            qwen_python,
            str(script),
            "--prompt",
            prompt,
            "--model",
            model,
            "--max-new-tokens",
            "256",
        ],
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Qwen subprocess failed with code {proc.returncode}: {proc.stderr[-2000:]}")
    return proc.stdout.strip()


def run_l1_to_rl_input(
    user_command: str,
    *,
    backend: str = "mock_qwen",
    output_path: str | Path | None = None,
    slot_map: str | Path | None = None,
    now_sec: float = 100.0,
    qwen_python: str = DEFAULT_QWEN_PYTHON,
    qwen_script: str = DEFAULT_QWEN_SCRIPT,
    qwen_timeout_s: float = 180.0,
) -> L1ClientResult:
    bridge = QwenMcpBridge(slot_map_path=slot_map, now_sec=now_sec)
    scene_context = bridge.call_tool("get_l1_scene_context", {"include_slot_poses": True})
    prompt = build_l1_prompt(user_command, scene_context)

    if backend == "mock_qwen":
        model_text = mock_qwen_decision(user_command, scene_context)
    elif backend == "qwen_subprocess":
        model_text = call_qwen_subprocess(
            prompt,
            qwen_python=qwen_python,
            qwen_script=qwen_script,
            timeout_s=qwen_timeout_s,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    tool_call = _extract_json_object(model_text)
    if tool_call.get("tool") != "resolve_intent_packet":
        raise ValueError("L1 model must call resolve_intent_packet")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("L1 tool call must include object arguments")

    intent_resolution = bridge.call_tool("resolve_intent_packet", arguments)
    skill_request = bridge.call_tool(
        "prepare_phase1_skill_request",
        {
            "intent_packet": intent_resolution["intent_packet"],
            "dry_run": True,
            "semantic_subtasks": intent_resolution.get("semantic_subtasks", []),
        },
    )
    result = L1ClientResult(
        backend=backend,
        user_command=user_command,
        prompt=prompt,
        model_text=model_text,
        tool_call=dict(tool_call),
        intent_resolution=intent_resolution,
        skill_request=skill_request,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen/LLM L1 -> RL input bridge")
    parser.add_argument("--command", required=True, help="Natural-language user command")
    parser.add_argument("--backend", choices=["mock_qwen", "qwen_subprocess"], default="mock_qwen")
    parser.add_argument("--output", required=True, help="Path to write result JSON")
    parser.add_argument("--slot-map", default=None)
    parser.add_argument("--now-sec", type=float, default=100.0)
    parser.add_argument("--qwen-python", default=DEFAULT_QWEN_PYTHON)
    parser.add_argument("--qwen-script", default=DEFAULT_QWEN_SCRIPT)
    parser.add_argument("--qwen-timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    result = run_l1_to_rl_input(
        args.command,
        backend=args.backend,
        output_path=args.output,
        slot_map=args.slot_map,
        now_sec=args.now_sec,
        qwen_python=args.qwen_python,
        qwen_script=args.qwen_script,
        qwen_timeout_s=args.qwen_timeout_s,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
