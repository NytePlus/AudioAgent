"""Image captioning API wrapper."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageCaptionResult:
    """Image caption result returned by the model API."""

    caption: str
    model: str
    image_path: str


class ImageCaptionerModel:
    """OpenAI-compatible client for image captioning."""

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

    def caption_image(
        self,
        image_path: str | Path,
        prompt: str = "Describe this image in detail.",
    ) -> ImageCaptionResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image_path)},
                    },
                ],
            }
        ]

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        caption = completion.choices[0].message.content or ""
        return ImageCaptionResult(caption=caption, model=self.model, image_path=str(image_path))

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            suffix = image_path.suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
            }.get(suffix, "image/png")

        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{image_base64}"
