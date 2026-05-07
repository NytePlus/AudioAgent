#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:-/data/dev_oracle_v1/history/multitask.jsonl}
OUTPUT=${OUTPUT:-pred_no_critic.txt}
LOG_DIR=${LOG_DIR:-logs/batch_api_asr/no_critic}
WORKERS=${WORKERS:-32}
RETRIES=${RETRIES:-1}
TIMEOUT=${TIMEOUT:-0}
LIMIT=${LIMIT:-}
MAX_STEPS=${MAX_STEPS:-10}
MAX_TOKENS=${MAX_TOKENS:-4096}
FRONTEND_MAX_TOKENS=${FRONTEND_MAX_TOKENS:-2048}
IMAGE_FIELD=${IMAGE_FIELD:-image}
QUESTION=${QUESTION:-'Transcribe what is being said and correct domain terms using the image. Return only valid JSON with exactly this schema: {"transcription": "<corrected transcript text>"}'}

cmd=(
  python -m audio_agent.examples.batch_api_asr
  --input "$INPUT"
  --output "$OUTPUT"
  --log-dir "$LOG_DIR"
  --ablation no-critic
  --question "$QUESTION"
  --max-steps "$MAX_STEPS"
  --max-tokens "$MAX_TOKENS"
  --workers "$WORKERS"
  --retries "$RETRIES"
  --timeout "$TIMEOUT"
  --history-field "history"
  --image-file "/data/dev_oracle_v1/slides/multitask.jsonl"
  --image-field "$IMAGE_FIELD"
  --frontend-max-tokens "$FRONTEND_MAX_TOKENS"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"
