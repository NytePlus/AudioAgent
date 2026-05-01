"""Utilities for extracting transcript text from model answers."""

from __future__ import annotations

import json
import re
from typing import Any


TRANSCRIPT_KEYS = ("pred", "transcript", "answer", "text")


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
        extracted = _extract_from_json(parsed)
        if extracted:
            return extracted

    return " ".join(text.strip().split())


def _extract_from_json(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in TRANSCRIPT_KEYS:
            field_value = value.get(key)
            if isinstance(field_value, str) and field_value.strip():
                return " ".join(field_value.strip().split())
        for field_value in value.values():
            extracted = _extract_from_json(field_value)
            if extracted:
                return extracted
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
