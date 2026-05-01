#!/usr/bin/env python3
"""Quick environment test for Image QA tool."""

from __future__ import annotations

import os
import sys


def test_imports() -> bool:
    print("Testing imports...")
    try:
        import openai

        print(f"  OK openai {openai.__version__}")
    except ImportError as exc:
        print(f"  FAIL openai: {exc}")
        return False

    try:
        from model import ImageQAModel

        _ = ImageQAModel
        print("  OK ImageQAModel imported successfully")
    except ImportError as exc:
        print(f"  FAIL ImageQAModel: {exc}")
        return False

    return True


def test_api_key() -> bool:
    print("\nTesting API configuration...")
    if os.environ.get("DASHSCOPE_API_KEY"):
        print("  OK DASHSCOPE_API_KEY is set")
    else:
        print("  WARN DASHSCOPE_API_KEY not set (required for actual API calls)")
    return True


def main() -> int:
    print("=" * 60)
    print("Image QA Environment Test")
    print("=" * 60)
    print()

    results = [
        ("Imports", test_imports()),
        ("API Configuration", test_api_key()),
    ]

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    sys.exit(main())
