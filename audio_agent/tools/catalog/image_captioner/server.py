#!/usr/bin/env python3
"""Image captioner MCP server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class ImageCaptionerServer:
    """MCP server exposing image captioning through an API model."""

    def __init__(self) -> None:
        self._initialized = False
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._default_model = os.environ.get("DEFAULT_MODEL", "qwen3-omni-flash")
        self._temperature = float(os.environ.get("TEMPERATURE", "0.05"))
        self._max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
        self._tools = [
            {
                "name": "image_caption",
                "description": (
                    "Generate a caption or visual description for an image using "
                    "Qwen Omni Flash via API."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file to caption.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional captioning instructions.",
                            "default": "Describe this image in detail.",
                        },
                    },
                    "required": ["image_path"],
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
                    "serverInfo": {"name": "image-captioner-server", "version": "1.0.0"},
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
            if tool_name != "image_caption":
                raise ValueError(f"Unknown tool: {tool_name}")
            result = self._caption(arguments)
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

    def _caption(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from model import ImageCaptionerModel

        image_path = arguments.get("image_path", "")
        prompt = arguments.get("prompt", "Describe this image in detail.")
        if not image_path:
            raise ValueError("image_path is required")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        model = ImageCaptionerModel(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._default_model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        result = model.caption_image(image_path=image_path, prompt=prompt)

        result_text = (
            f"Caption model: {result.model}\n"
            f"Image: {result.image_path}\n\n"
            f"{result.caption}"
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
    ImageCaptionerServer().run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
