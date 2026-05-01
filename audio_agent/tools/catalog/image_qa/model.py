"""Image question-answering API wrapper."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageQAResult:
    """Image QA result returned by the model API."""

    answer: str
    model: str
    image_path: str
    question: str


class ImageQAModel:
    """OpenAI-compatible client for image question answering."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3-omni-flash",
        temperature: float = 0.05,
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

    def answer_question(self, image_path: str | Path, question: str) -> ImageQAResult:
        """Answer a natural-language question about an image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not question or not question.strip():
            raise ValueError("question is required")

        prompt = (
            "Answer the user's question based only on the provided image. "
            "If the image does not contain enough evidence, say so clearly.\n\n"
            f"Question: {question.strip()}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(image_path)}},
                ],
            }
        ]

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = completion.choices[0].message.content or ""
        return ImageQAResult(
            answer=answer,
            model=self.model,
            image_path=str(image_path),
            question=question.strip(),
        )

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
            }.get(image_path.suffix.lower(), "image/png")

        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{image_base64}"
