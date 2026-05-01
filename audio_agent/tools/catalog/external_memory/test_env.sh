#!/bin/bash
# Quick environment test wrapper for External Memory tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python test_env.py "$@"
