#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:-/data/dev_oracle_v1/slides/multitask.jsonl}
OUTPUT=${OUTPUT:-pred_frontend_only.txt}
LOG_DIR=${LOG_DIR:-logs/batch_api_asr/frontend_only}
WORKERS=${WORKERS:-32}
RETRIES=${RETRIES:-1}
TIMEOUT=${TIMEOUT:-0}
LIMIT=${LIMIT:-}
MAX_TOKENS=${MAX_TOKENS:-4096}
FRONTEND_MAX_TOKENS=${FRONTEND_MAX_TOKENS:-2048}
QUESTION=${QUESTION:-'Transcribe what is being said. Return only valid JSON with exactly this schema: {"transcription": "<transcript text>"}'}

cmd=(
  python -m audio_agent.examples.batch_api_asr
  --input "$INPUT"
  --output "$OUTPUT"
  --log-dir "$LOG_DIR"
  --ablation frontend-only
  --question "$QUESTION"
  --max-tokens "$MAX_TOKENS"
  --workers "$WORKERS"
  --retries "$RETRIES"
  --timeout "$TIMEOUT"
  --history-field "history"
  --frontend-max-tokens "$FRONTEND_MAX_TOKENS"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"
