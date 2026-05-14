"""Local Ollama provider via litellm.

Uses ``qwen3:30b`` by default for text and ``qwen2.5vl:7b`` for vision —
both well-supported in Ollama and capable of multilingual OCR. The
``OLLAMA_HOST`` environment variable controls the base URL (default
``http://localhost:11434``); litellm reads it directly.

This is the recommended provider for sensitive documents: nothing leaves
the user's machine.
"""

from __future__ import annotations

from typing import ClassVar

from ._litellm import LiteLLMProvider


class OllamaProvider(LiteLLMProvider):
    name: ClassVar[str] = "ollama"
    default_text_model: ClassVar[str] = "ollama/qwen3:30b"
    default_vision_model: ClassVar[str | None] = "ollama/qwen2.5vl:7b"
