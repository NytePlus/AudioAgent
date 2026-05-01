"""Audio keyword/text verification API wrapper."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KeywordVerificationResult:
    """Result for verifying whether target text appears in an audio file."""

    exists: bool
    confidence: float
    evidence: str
    model: str
    audio_path: str
    target_text: str


class KeywordVerificationModel:
    """OpenAI-compatible client for audio-text existence verification."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3.5-omni-plus",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai not installed. Run: pip install openai") from exc

            if not self.api_key:
                raise RuntimeError("DASHSCOPE_API_KEY not set. Please provide API key.")

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def verify(self, audio_path: str | Path, target_text: str) -> KeywordVerificationResult:
        """Verify whether target text is present in the speech content."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not target_text or not target_text.strip():
            raise ValueError("target_text is required")

        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_format = audio_path.suffix.lstrip(".").lower()
        if audio_format not in ["wav", "mp3", "ogg", "m4a", "flac"]:
            audio_format = "wav"

        prompt = (
            "Listen to the audio and decide whether the target text is present in the "
            "spoken content. Match semantically equivalent wording only when it is clearly "
            "the same phrase or keyword. Return only valid JSON with this schema: "
            '{"exists": true|false, "confidence": 0.0-1.0, "evidence": "<brief reason>"}.\n\n'
            f"Target text: {target_text.strip()}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio_base64}",
                            "format": audio_format,
                        },
                    },
                ],
            }
        ]

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            modalities=["text"],
        )
        content = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        parsed = self._parse_json(content)
        return KeywordVerificationResult(
            exists=bool(parsed.get("exists", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            evidence=str(parsed.get("evidence", content)).strip(),
            model=self.model,
            audio_path=str(audio_path),
            target_text=target_text.strip(),
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return {"exists": False, "confidence": 0.0, "evidence": text}
        return data if isinstance(data, dict) else {}
