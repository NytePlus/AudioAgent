#!/usr/bin/env bash
# Lightweight setup for API-based inference on CPU/Mac environments.
#
# This intentionally skips local model/GPU tools that install torch or large
# speech models. It installs the core package with API support and only the
# lightweight MCP tools:
#   - external_memory (dummy retrieval)
#   - ffmpeg
#   - image_qa (API-based)
#   - image_captioner
#   - kw_verify (API-based)
#   - librosa
#   - omni_captioner (API-based)
#   - qwen3_asr_flash (API-based)
#   - qwen_vl_ocr (API-based)
#
# Usage:
#   ./light_setup.sh
#   ./light_setup.sh --verify
#   ./light_setup.sh --core-only
#   ./light_setup.sh --tools external_memory,image_qa,kw_verify --verify

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG_DIR="$SCRIPT_DIR/audio_agent/tools/catalog"
TOOLS=(external_memory ffmpeg image_qa image_captioner kw_verify librosa omni_captioner qwen3_asr_flash qwen_vl_ocr)
SELECTED_TOOLS=("${TOOLS[@]}")

VERIFY=false
CORE_ONLY=false
CONTINUE_ON_ERROR=false

usage() {
    cat <<'EOF'
Usage: ./light_setup.sh [options]

Options:
  --verify                 Run each selected tool's test_env.sh after setup.
  --core-only              Only install the root framework environment.
  --tools a,b,c            Setup only the comma-separated tool list.
  --continue-on-error      Continue setting up remaining tools if one fails.
  -h, --help               Show this help.

Lightweight tools:
  external_memory, ffmpeg, image_qa, image_captioner, kw_verify, librosa,
  omni_captioner, qwen3_asr_flash, qwen_vl_ocr
EOF
}

split_tools() {
    local tools_csv="$1"
    IFS=',' read -r -a SELECTED_TOOLS <<< "$tools_csv"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --verify)
            VERIFY=true
            shift
            ;;
        --core-only)
            CORE_ONLY=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --tools)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --tools" >&2
                usage
                exit 1
            fi
            split_tools "$2"
            shift 2
            ;;
        --tools=*)
            split_tools "${1#--tools=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

info() {
    echo "[light-setup] $*"
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi

    if [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
        return
    fi

    echo "[light-setup] uv not found; installing uv with the official installer" >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
        return
    fi

    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi

    echo "uv installation completed, but uv was not found on PATH." >&2
    echo "Try: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
}

find_python() {
    if command -v python3.11 >/dev/null 2>&1; then
        command -v python3.11
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi

    echo "Python 3.11+ is required, but python3 was not found." >&2
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

setup_core_env() {
    local uv_bin="$1"
    local python_bin="$2"

    info "Creating root .venv for the API-based framework"
    "$uv_bin" venv "$SCRIPT_DIR/.venv" --python "$python_bin"

    info "Installing audio_agent with API dependencies"
    "$uv_bin" pip install --python "$SCRIPT_DIR/.venv/bin/python" -e "${SCRIPT_DIR}[api]"

    info "Verifying API dependency: openai"
    "$SCRIPT_DIR/.venv/bin/python" -c "import openai; print(f'openai {openai.__version__}')"
}

setup_tool() {
    local tool_name="$1"
    local uv_bin="$2"
    local python_bin="$3"
    local tool_dir="$CATALOG_DIR/$tool_name"

    if [ ! -d "$tool_dir" ]; then
        echo "Tool directory not found: $tool_dir" >&2
        return 1
    fi

    if [ ! -f "$tool_dir/setup.sh" ]; then
        echo "Tool has no setup.sh: $tool_name" >&2
        return 1
    fi

    info "Setting up lightweight tool: $tool_name"
    (cd "$tool_dir" && UV="$uv_bin" PYTHON="$python_bin" bash setup.sh)
}

verify_tool() {
    local tool_name="$1"
    local tool_dir="$CATALOG_DIR/$tool_name"

    if [ ! -f "$tool_dir/test_env.sh" ]; then
        info "Skipping verification for $tool_name; no test_env.sh"
        return 0
    fi

    info "Verifying lightweight tool: $tool_name"
    (cd "$tool_dir" && bash test_env.sh)
}

main() {
    info "Starting lightweight setup for API/CPU environments"

    local uv_bin
    uv_bin="$(find_uv)"
    export PATH="$(dirname "$uv_bin"):$PATH"
    info "Using uv: $uv_bin"

    local python_bin
    python_bin="$(find_python)"
    check_python_version "$python_bin"
    info "Using Python: $("$python_bin" --version)"

    setup_core_env "$uv_bin" "$python_bin"

    local failed=0

    if [ "$CORE_ONLY" = false ]; then
        for tool_name in "${SELECTED_TOOLS[@]}"; do
            if ! setup_tool "$tool_name" "$uv_bin" "$python_bin"; then
                failed=$((failed + 1))
                if [ "$CONTINUE_ON_ERROR" = false ]; then
                    exit 1
                fi
            fi
        done

        if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
            info "ffmpeg/ffprobe command not found. On macOS, install them with: brew install ffmpeg"
        fi

        if [ "$VERIFY" = true ]; then
            for tool_name in "${SELECTED_TOOLS[@]}"; do
                if ! verify_tool "$tool_name"; then
                    failed=$((failed + 1))
                    if [ "$CONTINUE_ON_ERROR" = false ]; then
                        exit 1
                    fi
                fi
            done
        fi
    fi

    if [ "$failed" -gt 0 ]; then
        echo "[light-setup] Completed with $failed failure(s)" >&2
        exit 1
    fi

    info "Done"
    info "Activate the framework environment with: source .venv/bin/activate"
    info "For API runs, set DASHSCOPE_API_KEY before using API-based demos/tools."
}

main
