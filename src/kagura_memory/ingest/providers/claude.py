"""Anthropic Claude provider via litellm.

Uses Sonnet 4.6 by default for both text and vision (Sonnet has stronger
vision than Haiku, balancing cost and capability for ingestion). API key
is read from ``ANTHROPIC_API_KEY`` by litellm; the SDK does not store it.
"""

from __future__ import annotations

from typing import ClassVar

from ._litellm import LiteLLMProvider


class ClaudeProvider(LiteLLMProvider):
    name: ClassVar[str] = "claude"
    default_text_model: ClassVar[str] = "claude-sonnet-4-6"
    default_vision_model: ClassVar[str | None] = "claude-sonnet-4-6"
