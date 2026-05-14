"""LLM providers for the file ingestion pipeline.

Each provider implements :class:`Provider` and wraps ``litellm`` to talk to
its respective backend. Concrete providers (Claude, Gemini, Ollama) are
thin subclasses of a shared ``LiteLLMProvider`` base that handles the
litellm call dispatch.

Usage::

    from kagura_memory.ingest.providers import get_provider
    text_provider = get_provider("claude")
    summary = await text_provider.summarize("...long section...", max_tokens=200)

The :func:`get_provider` registry maps short names to constructors and is
the only entry point CLI/orchestrator code should use.
"""

from __future__ import annotations

from .base import Provider


def get_provider(name: str) -> Provider:
    """Resolve a short provider name to a :class:`Provider` instance.

    Names: ``"claude"``, ``"gemini"``, ``"ollama"``.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    name = name.lower()
    if name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider()
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider()
    raise ValueError(f"unknown provider {name!r}; choose from claude | gemini | ollama")


__all__ = ["Provider", "get_provider"]
