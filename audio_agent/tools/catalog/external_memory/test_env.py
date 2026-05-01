#!/usr/bin/env python3
"""Quick environment test for External Memory tool."""

from __future__ import annotations

import os
import sys


def main() -> int:
    print("=" * 60)
    print("External Memory Environment Test")
    print("=" * 60)
    print("  OK standard-library-only tool")
    if os.environ.get("EXTERNAL_MEMORY_PATH"):
        print("  OK EXTERNAL_MEMORY_PATH is set")
    elif os.environ.get("EXTERNAL_MEMORY_TEXT"):
        print("  OK EXTERNAL_MEMORY_TEXT is set")
    else:
        print("  WARN no external memory supplied; tool will return an empty memory string")
    return 0


if __name__ == "__main__":
    sys.exit(main())
