#!/usr/bin/env python3
"""Keyword verification MCP server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class KeywordVerifyServer:
    """MCP server exposing audio-text verification through an omni model."""

    def __init__(self) -> None:
        self._initialized = False
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._default_model = os.environ.get("DEFAULT_MODEL", "qwen3.5-omni-plus")
        self._temperature = float(os.environ.get("TEMPERATURE", "0.0"))
        self._max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
        self._tools = [
            {
                "name": "kw_verify",
                "description": (
                    "Use an omni audio model to verify whether a target text, keyword, "
                    "or phrase exists in the provided speech audio."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to inspect.",
                        },
                        "target_text": {
                            "type": "string",
                            "description": "Target text, keyword, or phrase to verify in the audio.",
                        },
                    },
                    "required": ["audio_path", "target_text"],
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
                    "serverInfo": {"name": "kw-verify-server", "version": "1.0.0"},
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
            if tool_name != "kw_verify":
                raise ValueError(f"Unknown tool: {tool_name}")
            result = self._verify(arguments)
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

    def _verify(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from model import KeywordVerificationModel

        audio_path = arguments.get("audio_path", "")
        target_text = arguments.get("target_text", "")
        if not audio_path:
            raise ValueError("audio_path is required")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not target_text:
            raise ValueError("target_text is required")

        model = KeywordVerificationModel(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._default_model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        result = model.verify(audio_path=audio_path, target_text=target_text)
        result_text = json.dumps(
            {
                "exists": result.exists,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "model": result.model,
                "audio_path": result.audio_path,
                "target_text": result.target_text,
            },
            ensure_ascii=False,
        )
        return {"content": [{"type": "text", "text": result_text}], "isError": False}

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
    KeywordVerifyServer().run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
