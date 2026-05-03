"""Utilities for extracting transcript text from model answers."""

from __future__ import annotations

import json
import re
from typing import Any


CANONICAL_TRANSCRIPTION_FORMAT = (
    'Return valid JSON containing a string "transcription" field. Extra fields are allowed, '
    'but only "transcription" is treated as the spoken transcript.'
)
TRANSCRIPT_KEYS = ("transcription", "corrected_transcription", "pred", "transcript", "answer", "text")
TRANSCRIPTION_TASK_MARKERS = (
    "transcribe",
    "transcription",
    "transcript",
    "asr",
    "speech-to-text",
    "speech to text",
    "what is being said",
)


def extract_transcript_text(text: str | None) -> str | None:
    """Extract likely transcript text from JSON-ish output, falling back to plain text."""
    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    json_text = _extract_balanced_json(candidate)
    if json_text:
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            return None
        extracted = _extract_from_json(parsed)
        if extracted:
            return extracted
        return None

    return " ".join(text.strip().split())


def looks_like_transcription_task(
    question: str | None,
    expected_output_format: str | None = None,
) -> bool:
    """Return True when the task should produce a transcript-shaped answer."""
    combined = f"{question or ''}\n{expected_output_format or ''}".casefold()
    return any(marker in combined for marker in TRANSCRIPTION_TASK_MARKERS)


def normalize_transcription_answer(text: str) -> str:
    """Normalize known transcript aliases to the canonical final-answer key."""
    transcript = extract_transcript_text(text)
    if not transcript:
        return text
    return json.dumps({"transcription": transcript}, ensure_ascii=False)


def _extract_from_json(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in TRANSCRIPT_KEYS:
            field_value = value.get(key)
            if isinstance(field_value, str) and field_value.strip():
                return " ".join(field_value.strip().split())
    if isinstance(value, list):
        parts = [_extract_from_json(item) for item in value]
        joined = " ".join(part for part in parts if part)
        return joined or None
    return None


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
