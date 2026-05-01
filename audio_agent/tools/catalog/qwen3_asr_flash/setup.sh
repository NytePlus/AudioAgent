#!/bin/bash
# Setup script for Qwen3-ASR-Flash API tool.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Setting up Qwen3-ASR-Flash API tool"
echo "============================================================"
echo ""

if [ -f "/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv" ]; then
    UV="/lihaoyu/workspace/AUDIO_AGENT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

echo "Using uv: $UV"
echo "uv version: $($UV --version)"
echo ""

if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

echo "Installing Qwen3-ASR-Flash API tool..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Set your DashScope API key before use:"
echo ""
echo "    export DASHSCOPE_API_KEY='your-api-key'"
echo ""
echo "To verify installation:"
echo "  ./test_env.sh"
