#!/usr/bin/env python3
"""MCP server exposing Qwen3-ASR-Flash API transcription."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class Qwen3ASRFlashServer:
    """MCP server for DashScope Qwen3-ASR-Flash."""

    def __init__(self) -> None:
        self._initialized = False
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._default_model = os.environ.get("DEFAULT_MODEL", "qwen3-asr-flash")
        self._default_language = os.environ.get("LANGUAGE", "auto")
        self._default_enable_itn = os.environ.get("ENABLE_ITN", "false").lower() == "true"
        self._default_stream = os.environ.get("STREAM", "false").lower() == "true"
        self._max_local_file_mb = int(os.environ.get("MAX_LOCAL_FILE_MB", "10"))
        self._tools = [
            {
                "name": "transcribe_qwen3_asr_flash",
                "description": (
                    "Transcribe speech with the qwen3-asr-flash API. Accepts an audio_path "
                    "or HTTP(S) audio URL and an optional context string for biasing "
                    "recognition toward domain terms, names, vocabulary, or reference text."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": (
                                "Path to a local audio file, an audio_id such as audio_0, "
                                "or a public HTTP(S) audio URL."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional context for recognition biasing, such as proper "
                                "nouns, slide/OCR text, terminology, or reference notes."
                            ),
                            "default": "",
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Optional language hint. Use values supported by Qwen-ASR "
                                "such as zh, en, ja, ko, or auto to omit the hint."
                            ),
                            "default": "auto",
                        },
                        "enable_itn": {
                            "type": "boolean",
                            "description": (
                                "Enable inverse text normalization for Chinese/English "
                                "spoken forms, such as converting 'one hundred' to '100'."
                            ),
                            "default": False,
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Use streaming API response collection.",
                            "default": False,
                        },
                    },
                    "required": ["audio_path"],
                },
            },
            {
                "name": "healthcheck",
                "description": "Verify qwen3_asr_flash server dependencies without calling the API.",
                "inputSchema": {"type": "object", "properties": {}},
            },
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
                    "serverInfo": {"name": "qwen3-asr-flash-server", "version": "1.0.0"},
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
            if tool_name == "transcribe_qwen3_asr_flash":
                result = self._transcribe(arguments)
            elif tool_name == "healthcheck":
                result = self._healthcheck()
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
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

    def _transcribe(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from model import Qwen3ASRFlashModel

        audio_path = arguments.get("audio_path", "")
        context = arguments.get("context", "")
        language = arguments.get("language", self._default_language)
        enable_itn = arguments.get("enable_itn", self._default_enable_itn)
        stream = arguments.get("stream", self._default_stream)

        if not audio_path:
            raise ValueError("audio_path is required")
        if not self._looks_like_url(audio_path) and not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = Qwen3ASRFlashModel(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._default_model,
            max_local_file_mb=self._max_local_file_mb,
        )
        result = model.transcribe(
            audio_path=audio_path,
            context=context,
            language=language,
            enable_itn=bool(enable_itn),
            stream=bool(stream),
        )

        payload = {
            "model": result.model,
            "audio_path": result.audio_source,
            "text": result.transcript,
            "language": result.language,
            # "enable_itn": result.enable_itn,
            # "context_used": result.context_used,
            # "annotations": result.annotations,
            # "usage": result.usage,
        }
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        }

    @staticmethod
    def _healthcheck() -> dict[str, Any]:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(f"Missing openai package: {exc}") from exc

        payload = {"status": "ok", "openai_version": openai.__version__}
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        }

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

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
    Qwen3ASRFlashServer().run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
