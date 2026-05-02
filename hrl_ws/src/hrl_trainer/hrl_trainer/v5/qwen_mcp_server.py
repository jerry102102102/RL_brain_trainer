"""Minimal stdio MCP server for Qwen-facing V5 tools.

The implementation is dependency-free on purpose: it supports the MCP methods
needed by common clients (`initialize`, `tools/list`, and `tools/call`) while the
tool logic lives in `qwen_mcp_tools.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping

from .qwen_mcp_tools import QwenMcpBridge, QwenMcpToolError


MCP_PROTOCOL_VERSION = "2024-11-05"


def _result(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _error(request_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _tool_content(payload: Mapping[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            }
        ],
        "isError": is_error,
    }


class QwenMcpStdioServer:
    def __init__(self, bridge: QwenMcpBridge):
        self.bridge = bridge

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if request_id is None and method in {"notifications/initialized", "notifications/cancelled"}:
            return None

        if method == "initialize":
            return _result(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "hrl-v5-qwen-mcp", "version": "0.1.0"},
                },
            )
        if method == "tools/list":
            return _result(request_id, {"tools": self.bridge.list_tools()})
        if method == "tools/call":
            if not isinstance(params, Mapping):
                return _error(request_id, -32602, "params must be an object")
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str):
                return _error(request_id, -32602, "tools/call requires string name")
            try:
                payload = self.bridge.call_tool(name, arguments)
                return _result(request_id, _tool_content(payload))
            except QwenMcpToolError as exc:
                return _result(request_id, _tool_content({"status": "error", "message": str(exc)}, is_error=True))
            except Exception as exc:  # pragma: no cover - defensive server boundary
                return _error(request_id, -32603, "Internal server error", {"detail": str(exc)})
        if method == "ping":
            return _result(request_id, {})
        return _error(request_id, -32601, f"Unknown method: {method}")

    def serve(self) -> int:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, Mapping):
                    raise ValueError("message must be a JSON object")
                response = self.handle(message)
            except Exception as exc:
                response = _error(None, -32700, "Parse error", {"detail": str(exc)})
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V5 Qwen MCP stdio server")
    parser.add_argument("--slot-map", default=None, help="Optional override path for v5_slot_map.yaml")
    parser.add_argument("--now-sec", type=float, default=100.0)
    parser.add_argument("--list-tools", action="store_true", help="Print tool manifest and exit")
    parser.add_argument("--call-tool", default=None, help="Call one tool once and exit")
    parser.add_argument("--arguments-json", default="{}", help="JSON object for --call-tool")
    args = parser.parse_args()

    bridge = QwenMcpBridge(slot_map_path=args.slot_map, now_sec=args.now_sec)
    if args.list_tools:
        print(json.dumps({"tools": bridge.list_tools()}, indent=2, sort_keys=True))
        return 0
    if args.call_tool:
        arguments = json.loads(args.arguments_json)
        print(json.dumps(bridge.call_tool(args.call_tool, arguments), indent=2, sort_keys=True))
        return 0
    return QwenMcpStdioServer(bridge).serve()


if __name__ == "__main__":
    raise SystemExit(main())

