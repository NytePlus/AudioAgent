#!/bin/bash
# Quick environment test for Qwen3-ASR-Flash API tool.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run ./setup.sh first."
    exit 1
fi

.venv/bin/python test_env.py
