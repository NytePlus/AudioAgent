#!/usr/bin/env python3
"""Quick environment test for the Qwen3-ASR-Flash API tool."""

from __future__ import annotations

import json
import os


def test_imports() -> bool:
    """Verify required Python packages and local modules import."""
    print("Testing imports...")
    try:
        import openai

        from model import Qwen3ASRFlashModel

        print(f"  openai: {openai.__version__}")
        print(f"  model wrapper: {Qwen3ASRFlashModel.__name__}")
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def test_server_healthcheck() -> bool:
    """Verify the server can execute its non-API healthcheck path."""
    print("\nTesting server healthcheck...")
    try:
        from server import Qwen3ASRFlashServer

        server = Qwen3ASRFlashServer()
        result = server._healthcheck()
        payload = json.loads(result["content"][0]["text"])
        print(f"  status: {payload['status']}")
        return payload["status"] == "ok"
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def test_api_key_visibility() -> bool:
    """Warn if API key is not currently configured."""
    print("\nChecking API key...")
    if os.environ.get("DASHSCOPE_API_KEY"):
        print("  DASHSCOPE_API_KEY is set.")
    else:
        print("  DASHSCOPE_API_KEY is not set; API calls will fail until it is provided.")
    return True


def main() -> int:
    print("Qwen3-ASR-Flash API Tool Environment Test")
    print("=" * 60)
    checks = [
        ("Imports", test_imports()),
        ("Server Healthcheck", test_server_healthcheck()),
        ("API Key Visibility", test_api_key_visibility()),
    ]
    print("\n" + "=" * 60)
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"FAILED checks: {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
