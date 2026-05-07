"""Google Gemini provider via litellm.

Uses Gemini 2.5 Flash (Lite for text, full Flash for vision) — fast,
inexpensive, strong at OCR-style image extraction. API key is read from
``GEMINI_API_KEY`` by litellm.
"""

from __future__ import annotations

from typing import ClassVar

from ._litellm import LiteLLMProvider


class GeminiProvider(LiteLLMProvider):
    name: ClassVar[str] = "gemini"
    default_text_model: ClassVar[str] = "gemini/gemini-2.5-flash-lite"
    default_vision_model: ClassVar[str | None] = "gemini/gemini-2.5-flash"
