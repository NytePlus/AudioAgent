#!/usr/bin/env bash
# Setup script for Image QA tool (API-based).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

find_uv() {
    if [ -n "${UV:-}" ]; then
        echo "$UV"
        return
    fi

    if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
        echo "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
        return
    fi

    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi

    if [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
        return
    fi

    echo "uv not found; installing uv with the official installer" >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
        return
    fi

    echo "Error: uv installation completed, but uv was not found." >&2
    exit 1
}

find_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
        return
    fi

    if command -v python3.11 >/dev/null 2>&1; then
        command -v python3.11
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi

    echo "Error: Python 3.11+ is required, but python3 was not found." >&2
    exit 1
}

check_python_version() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
PY
}

UV_BIN="$(find_uv)"
PYTHON_BIN="$(find_python)"
check_python_version "$PYTHON_BIN"

echo "Using uv: $UV_BIN"
echo "Using Python: $("$PYTHON_BIN" --version)"

if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

echo "Creating virtual environment..."
"$UV_BIN" venv --python "$PYTHON_BIN"

echo "Installing Image QA package..."
"$UV_BIN" pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo "Set DASHSCOPE_API_KEY before calling the image_qa tool."
