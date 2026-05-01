#!/usr/bin/env python3
"""Dummy external memory retrieval MCP server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class ExternalMemoryServer:
    """MCP server returning history transcripts supplied by the launcher."""

    def __init__(self) -> None:
        self._initialized = False
        self._memory_text = os.environ.get("EXTERNAL_MEMORY_TEXT", "")
        self._memory_path = os.environ.get("EXTERNAL_MEMORY_PATH", "")
        self._tools = [
            {
                "name": "external_memory_retrieve",
                "description": (
                    "Retrieve historical transcript memory supplied to this run. "
                    "The current dummy implementation returns the JSONL history text "
                    "passed by batch_api_asr/demo launch arguments."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional retrieval query; ignored by the dummy backend.",
                            "default": "",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Optional number of memories to return; ignored by the dummy backend.",
                            "default": 1,
                        },
                    },
                    "required": [],
                },
            }
        ]

    def run(self) -> None:
        """Run the JSON-RPC server loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as exc:
                self._send_error(None, -32700, f"Parse error: {exc}")
            except Exception as exc:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {exc}")

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            self._initialized = True
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "external-memory-server", "version": "1.0.0"},
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            if not self._initialized:
                return self._error_response(request_id, -32001, "Server not initialized")
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self._tools}}
        if method == "tools/call":
            return self._handle_tools_call(request_id, params)
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        return self._error_response(request_id, -32601, f"Method not found: {method}")

    def _handle_tools_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")

        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            if tool_name != "external_memory_retrieve":
                raise ValueError(f"Unknown tool: {tool_name}")
            result = self._retrieve(arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            error_msg = str(exc)
            print(f"Tool execution error: {error_msg}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                    "isError": True,
                    "error": error_msg,
                },
            }

    def _retrieve(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", ""))
        memory_text = self._load_memory_text()
        payload = {
            "query": query,
            "memory": memory_text,
            "source": self._memory_path or "EXTERNAL_MEMORY_TEXT",
            "dummy": True,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        }

    def _load_memory_text(self) -> str:
        if self._memory_path:
            path = Path(self._memory_path)
            if not path.exists():
                raise FileNotFoundError(f"External memory file not found: {path}")
            return path.read_text(encoding="utf-8")
        return self._memory_text

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _send_response(self, response: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        self._send_response(self._error_response(request_id, code, message))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    ExternalMemoryServer().run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
